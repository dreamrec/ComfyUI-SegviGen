"""
SegviGenPreprocess: background removal using BiRefNet from TRELLIS2.

Import path (after sys.path injection):
  from rembg.BiRefNet import BiRefNet

BiRefNet maintains its own internal model singleton — this is TRELLIS2's
implementation detail, not a SegviGen concern.
"""
import logging
import torch
import numpy as np
import comfy.model_management as mm

from .helpers import tensor_to_pil, pil_to_tensor, check_interrupt

log = logging.getLogger("segvigen")


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

    def preprocess(self, image: torch.Tensor, background_color: str = "black"):
        from rembg.BiRefNet import BiRefNet
        from PIL import Image

        check_interrupt()

        pil_img = tensor_to_pil(image)

        log.info("SegviGen: running BiRefNet background removal")
        birefnet = BiRefNet()
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
