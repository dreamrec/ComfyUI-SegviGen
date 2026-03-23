"""
SegviGen sampler nodes:
  - SegviGenFullSampler: automatic full-mesh segmentation
  - SegviGenInteractiveSampler: point-guided segmentation (added in Task 9)
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "guidance_strength": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "guidance_rescale": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Variance correction for CFG. 0.0 = off (recommended). "
                               "Increase slightly if outputs are oversaturated.",
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
        seed: int = 0,
        steps: int = 12,
        guidance_strength: float = 7.5,
        guidance_rescale: float = 0.0,
        guidance_interval_start: float = 0.6,
        guidance_interval_end: float = 0.9,
    ):
        import comfy.model_management as mm
        from core.pipeline import run_full_segmentation

        ckpt_path = _get_checkpoint_path()
        pb = make_progress(steps)

        def progress(step):
            pb.update(1)

        log.info(f"SegviGen: full segmentation (steps={steps}, cfg={guidance_strength})")

        result = run_full_segmentation(
            model_config=model_config,
            slat=slat,
            cond=conditioning,
            checkpoint_path=ckpt_path,
            seed=seed,
            steps=steps,
            guidance_strength=guidance_strength,
            guidance_rescale=guidance_rescale,
            guidance_interval_start=guidance_interval_start,
            guidance_interval_end=guidance_interval_end,
            progress_callback=progress,
            interrupt_check=check_interrupt,
        )

        mm.soft_empty_cache()
        return (result,)


class SegviGenInteractiveSampler:
    """Run point-guided interactive part segmentation."""

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
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 50}),
                "guidance_strength": ("FLOAT", {"default": 7.5, "min": 1.0, "max": 20.0, "step": 0.1}),
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
        from core.pipeline import Sampler, _add_noise, _decode_to_labels
        from core.interactive import Gen3DSegInteractive, encode_points_for_sampler
        import safetensors.torch
        from stages import get_flow_model

        ckpt_path = _get_checkpoint_path()
        pb = make_progress(steps)
        torch.manual_seed(seed)

        # Read voxel resolution from the source voxel or use default
        voxel_resolution = slat.get("voxel", {}).get("resolution", 64) if slat.get("voxel") else 64

        flow_model = get_flow_model(model_config)
        state_dict = safetensors.torch.load_file(ckpt_path)
        flow_model.load_state_dict(state_dict, strict=False)
        gen = Gen3DSegInteractive(flow_model, voxel_resolution=voxel_resolution)
        gen.eval()

        # VRAM management via ModelPatcher
        import comfy.model_patcher
        patcher = comfy.model_patcher.ModelPatcher(
            gen,
            load_device=mm.get_torch_device(),
            offload_device=mm.unet_offload_device(),
        )
        mm.load_models_gpu([patcher])

        sampler = Sampler(
            model=gen,
            steps=steps,
            guidance_strength=guidance_strength,
            guidance_rescale=guidance_rescale,
            guidance_interval=(guidance_interval_start, guidance_interval_end),
        )

        latent = slat["latent"]
        device = latent.feats.device if hasattr(latent, 'feats') else "cuda"
        pos_cond = conditioning.get("cond_1024", conditioning["cond_512"])
        neg_cond = conditioning["neg_cond"]

        # Encode points — clamped to voxel_resolution
        point_tensor = encode_points_for_sampler(
            points, voxel_resolution=voxel_resolution, max_points=10, device=str(device)
        )

        x_init = _add_noise(latent, seed)

        # Pass input_points through model_kwargs so it reaches
        # Gen3DSegInteractive.forward(input_points=...) without monkey-patching
        result_latent = sampler.sample(
            x_init=x_init,
            cond=pos_cond,
            neg_cond=neg_cond,
            progress_callback=lambda s: pb.update(1),
            interrupt_check=check_interrupt,
            model_kwargs={"input_points": point_tensor},
        )

        labels = _decode_to_labels(result_latent, slat)
        if labels.max() < 1:  # 0 = background; need at least label 1 for one part
            raise ValueError("Segmentation returned 0 parts — try a different seed or more points")

        mm.soft_empty_cache()

        return ({
            "latent": result_latent,
            "labels": labels,
            "voxel": slat.get("voxel"),
            "mesh": None,
        },)
