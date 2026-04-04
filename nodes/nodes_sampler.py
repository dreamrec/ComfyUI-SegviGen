"""
SegviGen sampler nodes:
  - SegviGenFullSampler: automatic full-mesh segmentation
  - SegviGenInteractiveSampler: point-guided segmentation
"""
import logging
import os

from .helpers import check_interrupt, make_progress

log = logging.getLogger("segvigen")

_SEGVIGEN_MODELS_DIR = None

# ── Interactive sampler cache ────────────────────────────────────────────────
# Avoids re-loading the model and re-normalizing tex_slat when only the
# click points change between runs. Keyed on (slat_id, conditioning_id).
_interactive_cache = {
    "key": None,
    "gen": None,
    "patcher": None,
    "shape_cond": None,
    "tex_slat_normed": None,
    "coords": None,
    "coords_np": None,
    "noise_ch": None,
    "voxel_resolution": None,
}


def _get_models_dir() -> str:
    global _SEGVIGEN_MODELS_DIR
    if _SEGVIGEN_MODELS_DIR is None:
        import folder_paths
        _SEGVIGEN_MODELS_DIR = os.path.join(folder_paths.models_dir, "segvigen")
        os.makedirs(_SEGVIGEN_MODELS_DIR, exist_ok=True)
    return _SEGVIGEN_MODELS_DIR


def _load_segvigen_flow_model(model_config: dict, ckpt_path: str):
    """
    Load SegviGen's SLatFlowModel from checkpoint and wrap it for ComfyUI.

    Supports both formats:
      - .safetensors (legacy Aero-Ex): keys are 'flow_model.*'
      - .ckpt (fenghora PyTorch Lightning): keys are in state_dict['gen3dseg.flow_model.*']

    The model is wrapped in ComfyCompatFlowModel so ComfyUI's ModelPatcher
    can set `model.device` without hitting the read-only @property.
    """
    import torch
    from core.pipeline import get_flow_model, ComfyCompatFlowModel

    flow_model = get_flow_model(model_config)

    if ckpt_path.endswith(".safetensors"):
        import safetensors.torch
        sd = safetensors.torch.load_file(ckpt_path, device="cpu")
        # Aero-Ex safetensors: keys are 'flow_model.*'
        stripped = {k[len("flow_model."):]: v
                    for k, v in sd.items() if k.startswith("flow_model.")}
    else:
        # fenghora .ckpt (PyTorch Lightning): keys are 'gen3dseg.flow_model.*'
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        sd = ckpt.get("state_dict", ckpt)
        stripped = {}
        for k, v in sd.items():
            if k.startswith("gen3dseg.flow_model."):
                stripped[k[len("gen3dseg.flow_model."):]] = v
            elif k.startswith("flow_model."):
                stripped[k[len("flow_model."):]] = v
        del ckpt

    flow_model.load_state_dict(stripped, strict=False)
    del sd, stripped
    flow_model.eval()
    return ComfyCompatFlowModel(flow_model)


def _get_interactive_checkpoint_path() -> str:
    """Return path to the interactive SegviGen checkpoint, downloading if needed."""
    from core.checkpoints import resolve_checkpoint
    return resolve_checkpoint("interactive_binary")


def _load_interactive_checkpoint(model_config: dict, ckpt_path: str):
    """
    Load the interactive segmentation checkpoint (PyTorch Lightning format).

    The .ckpt file contains:
      - gen3dseg.flow_model.* — same SLatFlowModel as full_seg.safetensors
      - gen3dseg.seg_embeddings.weight — [1, 1536] learned point embedding

    Returns:
        (flow_model, seg_embed_weight) — flow_model is bare SLatFlowModel
    """
    import torch
    from core.pipeline import get_flow_model

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt["state_dict"]

    # Strip 'gen3dseg.flow_model.' prefix → bare SLatFlowModel keys
    flow_sd = {}
    for k, v in sd.items():
        if k.startswith("gen3dseg.flow_model."):
            flow_sd[k[len("gen3dseg.flow_model."):]] = v

    flow_model = get_flow_model(model_config)
    flow_model.load_state_dict(flow_sd, strict=False)
    # The interactive checkpoint saves ALL weights in bf16, but TimestepEmbedder
    # always computes t_freq in float32 (hardcoded .float() in its forward).
    # The original SLatFlowModel.convert_to() only converts `blocks` to bf16,
    # leaving t_embedder in float32 — we replicate that here.
    flow_model.t_embedder.float()
    flow_model.eval()

    # Extract seg_embeddings weight
    seg_embed = sd["gen3dseg.seg_embeddings.weight"]  # [1, 1536]
    n_keys = len(flow_sd)

    del ckpt, sd, flow_sd
    log.info(f"SegviGen: loaded interactive checkpoint ({n_keys} flow keys + seg_embeddings {seg_embed.shape})")

    return flow_model, seg_embed


class SegviGenFullSampler:
    """Run automatic full-mesh part segmentation."""

    CATEGORY = "SegviGen"
    FUNCTION = "sample"
    RETURN_TYPES = ("SEGVIGEN_SEG_RESULT",)
    RETURN_NAMES = ("seg_result",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_config": ("TRELLIS2_MODEL_CONFIG",),
                "slat": ("SEGVIGEN_SLAT",),
                "conditioning": ("SEGVIGEN_COND",),
            },
            "optional": {
                "trimesh": ("TRIMESH", {
                    "tooltip": "Connect the original mesh so the preview is rendered on geometry.",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "guidance_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "guidance_rescale": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Variance correction for CFG. 0.0 = off (recommended).",
                }),
                "guidance_interval_start": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "guidance_interval_end": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    def sample(
        self,
        model_config: dict,
        slat: dict,
        conditioning: dict,
        trimesh=None,
        seed: int = 0,
        steps: int = 12,
        guidance_strength: float = 7.5,
        guidance_rescale: float = 0.0,
        guidance_interval_start: float = 0.6,
        guidance_interval_end: float = 0.9,
    ):
        import torch
        import numpy as np
        import comfy.model_management as mm
        import comfy.model_patcher
        from trellis2.modules import sparse as _sp
        from core.sampler import SegviGenFlowSampler
        from core.contracts import (
            build_segvigen_seg_result, MODE_FULL, MODE_FULL_2D_GUIDED,
            SOURCE_SHAPE_ONLY, TASK_FULL_2D_GUIDED, get_shape_slat,
        )

        check_interrupt()

        # ── Source guard: full mode requires real tex_slat ────────────────
        if slat.get("source") == SOURCE_SHAPE_ONLY:
            log.warning(
                "SegviGen: Full sampler received source='shape_only' (no tex_slat). "
                "Quality will be degraded. Use SegviGenVoxelEncode with conditioning."
            )

        # ── Load checkpoint via core/checkpoints ─────────────────────────
        # Switch checkpoint based on conditioning task_mode
        from core.checkpoints import resolve_checkpoint
        from core.contracts import TASK_FULL_2D_GUIDED
        task_mode = conditioning.get("task_mode", "standard")
        if task_mode == TASK_FULL_2D_GUIDED:
            ckpt_path = resolve_checkpoint("full_2d_guided")
            log.info("SegviGen full: using 2D-guided checkpoint (full_seg_w_2d_map.ckpt)")
        else:
            ckpt_path = resolve_checkpoint("full")

        device = mm.get_torch_device()
        dtype_str = model_config.get("dtype", "fp16")
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}[dtype_str]

        from core.interactive import Gen3DSegInteractive

        log.info(f"SegviGen: loading full flow model from {ckpt_path}")
        flow_model = _load_segvigen_flow_model(model_config, ckpt_path)

        # Wrap in Gen3DSegInteractive for 2N token interleaving.
        # Full mode passes input_points=None so seg_embeddings are unused,
        # but the wrapper is needed for the upstream 2N tex_slat interleaving
        # architecture that SLatFlowModel.forward() alone cannot do.
        dummy_seg_embed = torch.zeros(1, 1536)
        gen = Gen3DSegInteractive(flow_model._flow, dummy_seg_embed)
        gen = gen.to(device=device)
        gen.flow_model.convert_to(dtype)
        gen.eval()

        patcher = comfy.model_patcher.ModelPatcher(
            gen, load_device=device,
            offload_device=mm.unet_offload_device(),
        )
        mm.load_models_gpu([patcher])

        torch.manual_seed(seed)

        # ── Load normalization stats + rescale_t ─────────────────────────
        import json
        import folder_paths as _fp
        _pipeline_json = os.path.join(_fp.models_dir, "trellis2", "pipeline.json")
        _shape_mean = _shape_std = _tex_mean = _tex_std = None
        _rescale_t = 1.0
        if os.path.isfile(_pipeline_json):
            with open(_pipeline_json) as _f:
                _pcfg = json.load(_f).get("args", {})
            _sn = _pcfg.get("shape_slat_normalization", {})
            _tn = _pcfg.get("tex_slat_normalization", {})
            if _sn.get("mean") and _sn.get("std"):
                _shape_mean = torch.tensor(_sn["mean"], device=device, dtype=torch.float32)
                _shape_std  = torch.tensor(_sn["std"],  device=device, dtype=torch.float32)
            if _tn.get("mean") and _tn.get("std"):
                _tex_mean = torch.tensor(_tn["mean"], device=device, dtype=torch.float32)
                _tex_std  = torch.tensor(_tn["std"],  device=device, dtype=torch.float32)
            _rescale_t = float(_pcfg.get("rescale_t", 1.0))
            log.info(f"SegviGen full: loaded normalization stats (rescale_t={_rescale_t})")

        # ── Prepare shape_slat + coords ──────────────────────────────────
        slat_latent = get_shape_slat(slat)
        coords = slat_latent.coords
        noise_ch = gen.flow_model.out_channels
        cond_ch  = gen.flow_model.in_channels - noise_ch

        noise = _sp.SparseTensor(
            feats=torch.randn(len(coords), noise_ch, device=device, dtype=torch.float32),
            coords=coords,
        )

        # ── Normalise shape_slat for concat_cond ─────────────────────────
        raw_feats = slat_latent.feats.to(device=device, dtype=torch.float32)
        if _shape_mean is not None and cond_ch > 0:
            normed_feats = (raw_feats - _shape_mean) / _shape_std
        else:
            normed_feats = raw_feats
        shape_cond = _sp.SparseTensor(
            feats=normed_feats, coords=coords,
        ) if cond_ch > 0 else None

        # ── tex_slat: use real if available, else zero proxy ─────────────
        tex_slat_raw = slat.get("tex_slat")
        if tex_slat_raw is not None:
            raw_tex_feats = tex_slat_raw.feats.to(device=device, dtype=torch.float32)
            if _tex_mean is not None and _tex_std is not None:
                tex_normed_feats = (raw_tex_feats - _tex_mean) / _tex_std
            else:
                tex_normed_feats = raw_tex_feats
            tex_slat_normed = _sp.SparseTensor(
                feats=tex_normed_feats,
                coords=tex_slat_raw.coords.to(device),
            )
            log.info("SegviGen full: using real tex_slat")
        else:
            tex_slat_normed = _sp.SparseTensor(
                feats=torch.zeros(len(coords), noise_ch, device=device, dtype=torch.float32),
                coords=coords,
            )
            log.info("SegviGen full: using zero tex_proxy (no real tex_slat)")

        # ── Conditioning ─────────────────────────────────────────────────
        pos_cond = conditioning.get("cond_1024", conditioning["cond_512"])
        neg_cond = conditioning["neg_cond"]

        def _to_device(obj, dev, dt=None):
            if hasattr(obj, 'to'):
                kwargs = {"device": dev}
                if dt is not None:
                    kwargs["dtype"] = dt
                return obj.to(**kwargs)
            return obj

        pos_cond = _to_device(pos_cond, device, dtype)
        neg_cond = _to_device(neg_cond, device, dtype)

        # ── Sample with guidance_rescale + rescale_t ─────────────────────
        sampler_wrapper = SegviGenFlowSampler(sigma_min=1e-5)
        log.info(f"SegviGen full: sampling ({steps} steps, cfg={guidance_strength}, "
                 f"rescale={guidance_rescale}, rescale_t={_rescale_t})")

        result = sampler_wrapper.sample(
            gen, noise,
            cond=pos_cond,
            neg_cond=neg_cond,
            steps=steps,
            guidance_strength=guidance_strength,
            guidance_rescale=guidance_rescale,
            guidance_interval=(guidance_interval_start, guidance_interval_end),
            rescale_t=_rescale_t,
            concat_cond=shape_cond,
            tex_slats=tex_slat_normed,
        )
        seg_latent = result.samples

        # ── Decode labels via core/decode.py ─────────────────────────────
        from core.decode import decode_seg_result

        vr = (slat.get("voxel") or {}).get("resolution", 512)
        coords_np = seg_latent.coords[:, 1:].cpu().numpy().astype(np.int32)
        subs = slat.get("subs")

        is_guided = task_mode == TASK_FULL_2D_GUIDED
        decode_mode = "full_2d_guided" if is_guided else "full"
        result_mode = MODE_FULL_2D_GUIDED if is_guided else MODE_FULL

        labels, labels_source, decoded_voxels = decode_seg_result(
            seg_latent, subs, coords_np, vr,
            mode=decode_mode,
            grid_resolution=min(vr, 128),
        )

        mm.soft_empty_cache()
        return (build_segvigen_seg_result(
            output_tex_slat=seg_latent,
            decoded_tex_voxels=decoded_voxels,
            labels=labels,
            labels_source=labels_source,
            mode=result_mode,
            mesh=trimesh,
            voxel=slat.get("voxel"),
            source=slat.get("source"),
        ),)


class SegviGenInteractiveSampler:
    """Run point-guided interactive part segmentation.

    Uses the dedicated interactive_seg.ckpt checkpoint with proper point-token
    interleaving. Each click point triggers a separate binary inference run.
    The N binary masks are merged into a multi-label segmentation grid.
    """

    CATEGORY = "SegviGen"
    FUNCTION = "sample"
    RETURN_TYPES = ("SEGVIGEN_SEG_RESULT",)
    RETURN_NAMES = ("seg_result",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_config": ("TRELLIS2_MODEL_CONFIG",),
                "slat": ("SEGVIGEN_SLAT",),
                "conditioning": ("SEGVIGEN_COND",),
                "points": ("SEGVIGEN_POINTS",),
            },
            "optional": {
                "trimesh": ("TRIMESH", {
                    "tooltip": "Connect the original mesh so the preview is rendered on geometry.",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "guidance_strength": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "guidance_rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "guidance_interval_start": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "guidance_interval_end": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "allow_legacy_shape_only_fallback": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Allow degraded results when tex_slat is unavailable. "
                               "Off by default — faithful mode requires real tex_slat.",
                }),
            },
        }

    def sample(
        self,
        model_config: dict,
        slat: dict,
        conditioning: dict,
        points: list,
        trimesh=None,
        seed: int = 0,
        steps: int = 12,
        guidance_strength: float = 7.5,
        guidance_rescale: float = 0.0,
        guidance_interval_start: float = 0.6,
        guidance_interval_end: float = 0.9,
        allow_legacy_shape_only_fallback: bool = False,
    ):
        import torch
        import numpy as np
        import comfy.model_management as mm
        import comfy.model_patcher
        from trellis2.modules import sparse as _sp
        from core.interactive import (
            Gen3DSegInteractive, pack_point_tokens,
            decode_seg_from_base_color, _decode_via_kmeans,
        )
        from core.sampler import SegviGenFlowSampler

        check_interrupt()

        if not points or len(points) == 0:
            # First-run passthrough: the MeshPicker has already registered the
            # mesh in the browser UI at this point.  Return a no-op result so
            # downstream Export/Render nodes don't crash — the user can now open
            # the 3D picker, click points, and run again.
            log.info(
                "SegviGen: no points provided — mesh registered in picker. "
                "Click '🎯 Open 3D Picker' on the MeshPicker node, select "
                "the part(s) you want, then hit Run again."
            )
            from core.contracts import build_segvigen_seg_result, MODE_PREVIEW_PASSTHROUGH
            return (build_segvigen_seg_result(
                mode=MODE_PREVIEW_PASSTHROUGH,
                mesh=trimesh,
                voxel=slat.get("voxel"),
                source=slat.get("source"),
            ),)

        # ── Source guard (hard-error by default) ─────────────────────────
        from core.contracts import (
            validate_segvigen_slat, SOURCE_SHAPE_ONLY, SOURCE_BRIDGE_FULL,
        )
        if slat.get("source") == SOURCE_SHAPE_ONLY:
            if not allow_legacy_shape_only_fallback:
                raise ValueError(
                    "SegviGen: SLAT has source='shape_only' (no tex_slat).\n\n"
                    "This usually means SegviGenVoxelEncode failed to sample tex_slat.\n"
                    "Check the ComfyUI terminal/logs for the original error.\n\n"
                    "Common causes:\n"
                    "  - TRELLIS2 pipeline.json not found (check models/trellis2/)\n"
                    "  - VRAM exhaustion during tex flow model sampling\n"
                    "  - Incompatible TRELLIS2 version\n\n"
                    "To run with degraded quality anyway, enable\n"
                    "'allow_legacy_shape_only_fallback' in the node's advanced options."
                )
            log.warning(
                "SegviGen: SLAT has source='shape_only' — legacy fallback enabled. "
                "Interactive segmentation quality will be degraded.",
            )

        # ── Load interactive checkpoint (cached across point reruns) ────
        ckpt_path = _get_interactive_checkpoint_path()
        device = mm.get_torch_device()
        dtype_str = model_config.get("dtype", "fp16")
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}[dtype_str]

        cache_key = (ckpt_path, dtype_str)
        if _interactive_cache["key"] == cache_key and _interactive_cache["gen"] is not None:
            gen = _interactive_cache["gen"]
            patcher = _interactive_cache["patcher"]
            mm.load_models_gpu([patcher])
            log.info("SegviGen: reusing cached interactive model (points-only rerun)")
        else:
            log.info(f"SegviGen: loading interactive model from {ckpt_path}")
            flow_model, seg_embed = _load_interactive_checkpoint(
                model_config, ckpt_path
            )
            gen = Gen3DSegInteractive(flow_model, seg_embed)
            gen = gen.to(device=device)
            gen.flow_model.convert_to(dtype)
            gen.eval()

            patcher = comfy.model_patcher.ModelPatcher(
                gen, load_device=device,
                offload_device=mm.unet_offload_device(),
            )
            mm.load_models_gpu([patcher])

            _interactive_cache["key"] = cache_key
            _interactive_cache["gen"] = gen
            _interactive_cache["patcher"] = patcher

        torch.manual_seed(seed)

        # ── Load normalization stats + rescale_t ─────────────────────────
        import json
        import folder_paths as _fp
        _pipeline_json = os.path.join(_fp.models_dir, "trellis2", "pipeline.json")
        _shape_mean = _shape_std = _tex_mean = _tex_std = None
        _rescale_t = 1.0
        if os.path.isfile(_pipeline_json):
            with open(_pipeline_json) as _f:
                _pcfg = json.load(_f).get("args", {})
            _sn = _pcfg.get("shape_slat_normalization", {})
            _tn = _pcfg.get("tex_slat_normalization", {})
            if _sn.get("mean") and _sn.get("std"):
                _shape_mean = torch.tensor(_sn["mean"], device=device, dtype=torch.float32)
                _shape_std  = torch.tensor(_sn["std"],  device=device, dtype=torch.float32)
            if _tn.get("mean") and _tn.get("std"):
                _tex_mean = torch.tensor(_tn["mean"], device=device, dtype=torch.float32)
                _tex_std  = torch.tensor(_tn["std"],  device=device, dtype=torch.float32)
            _rescale_t = float(_pcfg.get("rescale_t", 1.0))
            log.info("SegviGen: loaded normalization stats from pipeline.json "
                     f"(rescale_t={_rescale_t})")
        else:
            log.warning(
                "SegviGen: pipeline.json not found — running WITHOUT shape normalization. "
                "Place TRELLIS2 models in models/trellis2/ for best results."
            )

        # ── Prepare shared state ─────────────────────────────────────────
        slat_latent = slat["latent"]
        voxel_resolution = (slat.get("voxel") or {}).get("resolution", 512)
        coords = slat_latent.coords
        coords_np = coords[:, 1:].cpu().numpy().astype(np.int32)

        noise_ch = gen.flow_model.out_channels   # 32
        cond_ch = gen.flow_model.in_channels - noise_ch  # 32

        pos_cond = conditioning.get("cond_1024", conditioning["cond_512"])
        neg_cond = conditioning["neg_cond"]

        def _to_device(obj, dev, dt=None):
            """Move obj to device if it supports .to(), otherwise return as-is."""
            if hasattr(obj, 'to'):
                kwargs = {"device": dev}
                if dt is not None:
                    kwargs["dtype"] = dt
                return obj.to(**kwargs)
            return obj

        pos_cond = _to_device(pos_cond, device, dtype)
        neg_cond = _to_device(neg_cond, device, dtype)

        sampler_wrapper = SegviGenFlowSampler(sigma_min=1e-5)

        # ── Normalise shape_slat for concat_cond ─────────────────────────
        raw_shape_feats = slat_latent.feats.to(device=device, dtype=torch.float32)
        if _shape_mean is not None and _shape_std is not None:
            shape_normed_feats = (raw_shape_feats - _shape_mean) / _shape_std
            log.info(
                f"SegviGen: shape_slat normalised — "
                f"norm_mean={shape_normed_feats.mean():.4f} "
                f"norm_std={shape_normed_feats.std():.4f}"
            )
        else:
            shape_normed_feats = raw_shape_feats
        shape_cond = _sp.SparseTensor(
            feats=shape_normed_feats,
            coords=coords,
        ) if cond_ch > 0 else None

        # ── tex_slat: use real tex_slat if available, else zero proxy ────
        tex_slat_raw = slat.get("tex_slat")
        if tex_slat_raw is not None:
            # Re-normalize: _sample_tex_slat returns denormalized (feats * std + mean)
            # The sampler expects normalized values
            raw_tex_feats = tex_slat_raw.feats.to(device=device, dtype=torch.float32)
            if _tex_mean is not None and _tex_std is not None:
                tex_normed_feats = (raw_tex_feats - _tex_mean) / _tex_std
            else:
                tex_normed_feats = raw_tex_feats
            tex_slat_normed = _sp.SparseTensor(
                feats=tex_normed_feats,
                coords=tex_slat_raw.coords.to(device),
            )
            log.info("SegviGen: using real tex_slat for 2N interleaving")
        else:
            # Fallback: zero-filled proxy (degraded quality)
            tex_slat_normed = _sp.SparseTensor(
                feats=torch.zeros(len(coords), noise_ch, device=device, dtype=torch.float32),
                coords=coords,
            )
            log.info("SegviGen: using zero tex_proxy (no real tex_slat available)")

        # ── Map points to model coordinates ──────────────────────────────
        from core.contracts import SOURCE_ASSET_FULL
        if slat.get("source") == SOURCE_ASSET_FULL:
            # Asset-native: use tex_encoder for paper-faithful mapping
            try:
                from core.asset_encode import map_points_via_tex_encoder
                mapped_coords = map_points_via_tex_encoder(
                    points, coords, device=str(device),
                )
                # Build input_points from encoder-derived coords
                MAX = 10
                n = min(len(mapped_coords), MAX)
                pt_coords = torch.zeros(MAX, 4, dtype=torch.int32, device=device)
                pt_labels = torch.zeros(MAX, 1, dtype=torch.int32, device=device)
                pt_coords[:n] = mapped_coords[:n].to(dtype=torch.int32)
                pt_labels[:n] = 1
                input_pts = {'coords': pt_coords, 'labels': pt_labels}
                log.info(f"SegviGen: mapped {len(points)} points via tex_encoder")
            except Exception as e:
                log.warning(f"SegviGen: tex_encoder mapping failed ({e}); "
                            "falling back to nearest-neighbor snap")
                input_pts = pack_point_tokens(
                    points, voxel_resolution, device=str(device),
                    slat_coords=coords,
                )
        else:
            # Bridge mode: snap to nearest occupied SLAT coordinate
            input_pts = pack_point_tokens(
                points, voxel_resolution, device=str(device),
                slat_coords=coords,
            )
        log.info(f"SegviGen interactive: {len(points)} points packed, "
                 f"{steps} steps, cfg={guidance_strength}")

        # ── Single forward pass with all points ──────────────────────────
        noise = _sp.SparseTensor(
            feats=torch.randn(len(coords), noise_ch,
                              device=device, dtype=torch.float32),
            coords=coords,
        )

        result = sampler_wrapper.sample(
            gen, noise,
            cond=pos_cond, neg_cond=neg_cond,
            steps=steps,
            guidance_strength=guidance_strength,
            guidance_rescale=guidance_rescale,
            guidance_interval=(guidance_interval_start, guidance_interval_end),
            rescale_t=_rescale_t,
            input_points=input_pts,
            concat_cond=shape_cond,
            tex_slats=tex_slat_normed,
        )

        # ── Label extraction via core/decode.py ────────────────────────
        from core.decode import decode_seg_result
        from core.contracts import build_segvigen_seg_result, MODE_INTERACTIVE_BINARY

        subs = slat.get("subs")
        grid_res = min(voxel_resolution, 128)

        labels, labels_source, decoded_voxels = decode_seg_result(
            result.samples, subs, coords_np, voxel_resolution,
            mode="interactive_binary",
            grid_resolution=grid_res,
        )

        mm.soft_empty_cache()
        return (build_segvigen_seg_result(
            output_tex_slat=result.samples,
            decoded_tex_voxels=decoded_voxels,
            labels=labels,
            labels_source=labels_source,
            mode=MODE_INTERACTIVE_BINARY,
            mesh=trimesh,
            voxel=slat.get("voxel"),
            source=slat.get("source"),
        ),)
