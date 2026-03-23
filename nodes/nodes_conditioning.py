"""
SegviGen conditioning nodes:
  - SegviGenGetConditioning: DINOv3 visual features from an image+mask
  - SegviGenNullConditioning: null embedding for unconditioned auto-segmentation

Import: `from stages import run_conditioning`
(not `from trellis2.stages` — two stages.py files exist in the injected path;
 the correct one is ComfyUI-TRELLIS2/nodes/stages.py)

Returns SEGVIGEN_COND dict with keys:
  {
    "cond_512":  torch.Tensor,
    "neg_cond":  torch.Tensor,
    "cond_1024": torch.Tensor,  # present when resolution in 1024_cascade/1536_cascade/1024
  }
"""
import logging

from .helpers import check_interrupt

log = logging.getLogger("segvigen")


class SegviGenGetConditioning:
    """Extract DinoV3 visual features for conditioning the segmentation flow model."""

    CATEGORY = "SegviGen"
    FUNCTION = "condition"
    RETURN_TYPES = ("SEGVIGEN_COND",)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_config": ("TRELLIS2_MODEL_CONFIG", {
                    "tooltip": "Config from Load TRELLIS2 Models node",
                }),
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    def condition(self, model_config: dict, image, mask):
        import torch  # noqa: F401 — needed by stages.run_conditioning
        import comfy.model_management as mm
        from stages import run_conditioning  # ComfyUI-TRELLIS2/nodes/stages.py

        check_interrupt()

        resolution = model_config.get("resolution", "1024_cascade")
        include_1024 = resolution in ("1024_cascade", "1536_cascade", "1024")

        log.info(f"SegviGen: extracting DinoV3 conditioning (resolution={resolution})")

        cond, _ = run_conditioning(
            model_config=model_config,
            image=image,
            mask=mask,
            include_1024=include_1024,
            background_color="black",
        )

        mm.soft_empty_cache()

        return (cond,)


class SegviGenNullConditioning:
    """
    Null / unconditioned mode — no reference image required.

    Runs DINOv3 on a blank black image to obtain the model's own null embedding,
    then uses it as *both* the positive and negative conditioning branches.

    Because pos_cond == neg_cond, the CFG term cancels to zero regardless of
    guidance_strength:
        v_guided = v_uncond + strength × (v_cond − v_uncond)
                 = v_uncond + strength × 0
                 = v_uncond

    The model then segments based purely on its learned 3D structure priors,
    with no visual reference guiding which parts to separate.
    """

    CATEGORY = "SegviGen"
    FUNCTION = "make_null"
    RETURN_TYPES = ("SEGVIGEN_COND",)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_config": ("TRELLIS2_MODEL_CONFIG", {
                    "tooltip": "Config from Load TRELLIS2 Models node",
                }),
            },
        }

    def make_null(self, model_config: dict):
        import torch
        import comfy.model_management as mm
        from stages import run_conditioning  # ComfyUI-TRELLIS2/nodes/stages.py

        check_interrupt()

        resolution = model_config.get("resolution", "1024_cascade")
        include_1024 = resolution in ("1024_cascade", "1536_cascade", "1024")

        # Blank 512×512 black image — standard conditioning resolution.
        # ComfyUI IMAGE tensors are [B, H, W, C] float32 in [0, 1].
        blank_image = torch.zeros(1, 512, 512, 3)
        blank_mask = torch.ones(1, 512, 512)   # full white mask (no cropping)

        log.info("SegviGen: building null conditioning (blank image → DINOv3 null embedding)")

        cond_dict, _ = run_conditioning(
            model_config=model_config,
            image=blank_image,
            mask=blank_mask,
            include_1024=include_1024,
            background_color="black",
        )

        # Use neg_cond as both branches so CFG cancels to zero.
        null_embed = cond_dict["neg_cond"]
        null_cond = {
            "cond_512":  null_embed,
            "neg_cond":  null_embed,
        }
        if include_1024:
            null_cond["cond_1024"] = null_embed

        mm.soft_empty_cache()

        return (null_cond,)
