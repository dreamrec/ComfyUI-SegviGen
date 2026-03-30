"""
SegviGenPreprocess: background removal using BiRefNet from TRELLIS2.

TRELLIS2 ships its own rembg/BiRefNet.py inside its nodes/ directory —
it is NOT the standard pip 'rembg' package.  We locate it at runtime by
scanning sys.path for any directory that contains rembg/BiRefNet.py and
load it directly with importlib, avoiding any dependency on package naming.
"""
import logging
import importlib.util
import os
import sys
import numpy as np

from .helpers import tensor_to_pil, pil_to_tensor, check_interrupt

log = logging.getLogger("segvigen")


def _load_birefnet():
    """Return TRELLIS2's BiRefNet class, loading it from disk on first call."""
    for p in sys.path:
        if not p:
            continue
        candidate = os.path.join(p, "rembg", "BiRefNet.py")
        if os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location(
                "_segvigen_trellis2_birefnet", candidate
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            log.info(f"SegviGen: loaded BiRefNet from {candidate}")
            return mod.BiRefNet
    raise ImportError(
        "SegviGen: cannot find TRELLIS2's rembg/BiRefNet.py on sys.path.\n"
        "Make sure ComfyUI-TRELLIS2 is installed and its nodes/ directory "
        "is on sys.path."
    )


class SegviGenPreprocess:
    """Remove background from an image using BiRefNet, producing image + mask."""

    CATEGORY = "SegviGen"
    FUNCTION = "preprocess"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "background_color": (
                    ["black", "gray", "white"],
                    {"default": "black",
                     "tooltip": "Background fill color applied after mask."},
                ),
            },
        }

    def preprocess(self, image, background_color: str = "black"):
        import torch
        import comfy.model_management as mm
        from PIL import Image

        check_interrupt()

        pil_img = tensor_to_pil(image)

        log.info("SegviGen: running BiRefNet background removal")
        birefnet = _load_birefnet()()
        rgba_result = birefnet(pil_img)  # returns RGBA PIL image

        # Extract alpha as mask
        r, g, b, a = rgba_result.split()
        mask_np = np.array(a).astype(np.float32) / 255.0  # [H, W] 0-1

        # Apply background color
        bg_colors = {"black": (0, 0, 0), "gray": (128, 128, 128), "white": (255, 255, 255)}
        bg = Image.new("RGB", rgba_result.size, bg_colors[background_color])
        bg.paste(rgba_result, mask=a)

        out_image = pil_to_tensor(bg)
        out_mask = torch.from_numpy(mask_np).unsqueeze(0)  # [1, H, W]

        mm.soft_empty_cache()

        return (out_image, out_mask)
