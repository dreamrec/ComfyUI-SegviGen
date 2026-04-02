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


def _get_models_dir() -> str:
    global _SEGVIGEN_MODELS_DIR
    if _SEGVIGEN_MODELS_DIR is None:
        import folder_paths
        _SEGVIGEN_MODELS_DIR = os.path.join(folder_paths.models_dir, "segvigen")
        os.makedirs(_SEGVIGEN_MODELS_DIR, exist_ok=True)
    return _SEGVIGEN_MODELS_DIR


def _get_checkpoint_path() -> str:
    """Return path to the SegviGen checkpoint, downloading if needed."""
    from install import ensure_checkpoint
    return ensure_checkpoint(_get_models_dir())


def _load_segvigen_flow_model(model_config: dict, ckpt_path: str):
    """
    Load SegviGen's SLatFlowModel from checkpoint and wrap it for ComfyUI.

    The checkpoint was saved from a Gen3DSeg wrapper, so all keys have a
    'flow_model.' prefix that must be stripped before loading into the bare
    SLatFlowModel.  The model is then wrapped in ComfyCompatFlowModel so
    ComfyUI's ModelPatcher can set `model.device` without hitting the
    read-only @property on SLatFlowModel.
    """
    import safetensors.torch
    from core.pipeline import get_flow_model, ComfyCompatFlowModel

    flow_model = get_flow_model(model_config)
    sd = safetensors.torch.load_file(ckpt_path, device="cpu")
    stripped = {k[len("flow_model."):]: v
                for k, v in sd.items() if k.startswith("flow_model.")}
    flow_model.load_state_dict(stripped, strict=False)
    del sd, stripped
    flow_model.eval()
    return ComfyCompatFlowModel(flow_model)


def _get_interactive_checkpoint_path() -> str:
    """Return path to the interactive SegviGen checkpoint, downloading if needed."""
    from install import ensure_interactive_checkpoint
    return ensure_interactive_checkpoint(_get_models_dir())


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

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
        import comfy.model_management as mm
        import comfy.model_patcher
        from trellis2.samplers import FlowEulerGuidanceIntervalSampler
        from trellis2.modules import sparse as _sp  # internal SparseTensor recognised by model

        check_interrupt()
        pb = make_progress(steps)

        ckpt_path = _get_checkpoint_path()
        device = mm.get_torch_device()
        dtype_str = model_config.get("dtype", "fp16")
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]

        log.info(f"SegviGen: loading flow model from {ckpt_path}")
        flow_model = _load_segvigen_flow_model(model_config, ckpt_path)

        patcher = comfy.model_patcher.ModelPatcher(
            flow_model,
            load_device=device,
            offload_device=mm.unet_offload_device(),
        )
        mm.load_models_gpu([patcher])

        torch.manual_seed(seed)

        # ── Load normalization stats ──────────────────────────────────────
        import json
        import folder_paths as _fp
        _pipeline_json = os.path.join(_fp.models_dir, "trellis2", "pipeline.json")
        _shape_mean = _shape_std = None
        if os.path.isfile(_pipeline_json):
            with open(_pipeline_json) as _f:
                _pcfg = json.load(_f).get("args", {})
            _sn = _pcfg.get("shape_slat_normalization", {})
            if _sn.get("mean") and _sn.get("std"):
                _shape_mean = torch.tensor(_sn["mean"], device=device, dtype=torch.float32)
                _shape_std  = torch.tensor(_sn["std"],  device=device, dtype=torch.float32)

        slat_latent = slat["latent"]
        coords = slat_latent.coords
        noise_ch = flow_model.out_channels
        cond_ch  = flow_model.in_channels - flow_model.out_channels

        noise = _sp.SparseTensor(
            feats=torch.randn(len(coords), noise_ch, device=device, dtype=torch.float32),
            coords=coords,
        )

        # Normalise shape_slat for concat_cond
        raw_feats = slat_latent.feats.to(device=device, dtype=torch.float32)
        if _shape_mean is not None and cond_ch > 0:
            normed_feats = (raw_feats - _shape_mean) / _shape_std
        else:
            normed_feats = raw_feats
        concat_cond = _sp.SparseTensor(
            feats=normed_feats, coords=coords
        ) if cond_ch > 0 else None

        # tex_proxy: zeros at same coords (2N interleaving for FullSampler
        # uses ComfyCompatFlowModel which doesn't implement 2N — pass None)
        # The FullSampler wraps the bare SLatFlowModel (not Gen3DSegInteractive),
        # so we keep the existing N-token path for it.

        pos_cond = conditioning.get("cond_1024", conditioning["cond_512"])
        neg_cond = conditioning["neg_cond"]

        def _to_device(c):
            if isinstance(c, torch.Tensor):
                return c.to(device=device, dtype=dtype)
            if isinstance(c, list):
                return [t.to(device=device, dtype=dtype) if isinstance(t, torch.Tensor) else t for t in c]
            return c
        pos_cond = _to_device(pos_cond)
        neg_cond = _to_device(neg_cond)

        extra_model_kwargs = {"concat_cond": concat_cond} if concat_cond is not None else {}

        log.info(f"SegviGen: sampling ({steps} steps, cfg={guidance_strength})")
        sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
        result = sampler.sample(
            flow_model, noise,
            cond=pos_cond,
            neg_cond=neg_cond,
            steps=steps,
            guidance_strength=guidance_strength,
            guidance_interval=(guidance_interval_start, guidance_interval_end),
            verbose=True,
            tqdm_desc="SegviGen",
            **extra_model_kwargs,
        )
        seg_latent = result.samples

        # ── decode labels from model output via K-means clustering ────────
        import numpy as np
        from sklearn.cluster import MiniBatchKMeans

        vr = (slat.get("voxel") or {}).get("resolution", 512)
        seg_feats = seg_latent.feats.cpu().float().numpy()
        seg_coords_np = seg_latent.coords[:, 1:].cpu().numpy().astype(np.int32)

        # Auto-detect number of parts (default 4: e.g. turret / hull / tracks / other)
        n_parts = 4
        k = min(n_parts, max(2, len(seg_feats) // 10))
        km = MiniBatchKMeans(n_clusters=k, n_init=5, random_state=0)
        cluster_ids = km.fit_predict(seg_feats)

        # Build [G,G,G] label grid (1-based: cluster 0 → label 1, etc.)
        # Scale coords from vr (e.g. 512) down to 64-res to avoid OOM.
        G = min(vr, 64)
        scale = vr / G
        labels = np.zeros((G, G, G), dtype=np.int32)
        for j, (x, y, z) in enumerate(seg_coords_np):
            gx = min(int(x / scale), G - 1)
            gy = min(int(y / scale), G - 1)
            gz = min(int(z / scale), G - 1)
            if gx >= 0 and gy >= 0 and gz >= 0:
                labels[gx, gy, gz] = int(cluster_ids[j]) + 1
        log.info(f"SegviGen full: {k}-cluster K-means → "
                 f"{len(np.unique(cluster_ids))} segments (grid {G}³)")

        mm.soft_empty_cache()
        return ({"latent": seg_latent, "labels": labels, "voxel": slat.get("voxel"), "mesh": trimesh},)


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
            return ({"latent": None, "labels": None,
                     "voxel": slat.get("voxel"), "mesh": trimesh},)

        # ── Soft source guard ────────────────────────────────────────────
        if slat.get("source") != "full":
            log.warning(
                "SegviGen: SLAT has source='%s' (no tex_slat). "
                "Interactive segmentation quality will be degraded. "
                "For best results, connect SegviGenVoxelEncode with conditioning.",
                slat.get("source", "unknown"),
            )

        # ── Load interactive checkpoint ──────────────────────────────────
        ckpt_path = _get_interactive_checkpoint_path()
        device = mm.get_torch_device()
        dtype_str = model_config.get("dtype", "fp16")
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}[dtype_str]

        log.info(f"SegviGen: loading interactive model from {ckpt_path}")
        flow_model, seg_embed = _load_interactive_checkpoint(
            model_config, ckpt_path
        )

        # Create Gen3DSegInteractive with token interleaving.
        # convert_to() mirrors SLatFlowModel's own __init__: only the 30 transformer
        # blocks are converted to bf16/fp16; input_layer, t_embedder, out_layer, and
        # pos_embedder stay in float32 so they match the float32 inputs/outputs of the
        # sampler.
        gen = Gen3DSegInteractive(flow_model, seg_embed)
        gen = gen.to(device=device)
        gen.flow_model.convert_to(dtype)   # blocks → bf16/fp16, fm.dtype updated
        gen.eval()

        # Use ModelPatcher for VRAM management
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

        # ── Pack all points into single input_points dict ────────────────
        input_pts = pack_point_tokens(points, voxel_resolution, device=str(device))
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

        # ── Label extraction ─────────────────────────────────────────────
        subs = slat.get("subs")
        grid_res = min(voxel_resolution, 64)
        if subs is not None:
            labels = decode_seg_from_base_color(
                result.samples, subs,
                coords_np, voxel_resolution,
                grid_resolution=grid_res,
            )
        else:
            labels = _decode_via_kmeans(
                result.samples, coords_np, grid_res,
            )

        mm.soft_empty_cache()
        return ({"latent": result.samples, "labels": labels,
                 "voxel": slat.get("voxel"), "mesh": trimesh},)
