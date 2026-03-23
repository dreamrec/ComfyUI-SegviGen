"""
SegviGenGetConditioning: DinoV3 feature extraction via TRELLIS2's stages.

Import: `from stages import run_conditioning`
(not `from trellis2.stages` — two stages.py files exist in the injected path;
 the correct one is ComfyUI-TRELLIS2/nodes/stages.py)

Returns SEGVIGEN_COND dict with keys matching run_conditioning output:
  {
    "cond_512": torch.Tensor,
    "neg_cond": torch.Tensor,
    "cond_1024": torch.Tensor,  # present when resolution in 1024_cascade/1536_cascade/1024
  }
"""
import logging
import torch
import comfy.model_management as mm

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

    def condition(self, model_config: dict, image: torch.Tensor, mask: torch.Tensor):
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
