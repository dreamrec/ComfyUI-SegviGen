"""
Shared helper utilities for ComfyUI-SegviGen nodes.
"""
import logging
import numpy as np
import torch
from PIL import Image

log = logging.getLogger("segvigen")


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert ComfyUI IMAGE tensor [B,H,W,C] or [H,W,C] float32 → PIL RGB.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Convert PIL Image → ComfyUI IMAGE tensor [1,H,W,C] float32 0-1.
    RGBA is converted to RGB (alpha channel discarded).
    """
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def check_interrupt():
    """
    Call inside long loops to allow ComfyUI to cancel the current execution.
    Raises comfy.model_management.InterruptProcessingException if cancelled.
    """
    import comfy.model_management as mm
    mm.throw_exception_if_processing_interrupted()


def make_progress(total: int):
    """
    Return a ComfyUI ProgressBar with .update(n) interface.

    Usage:
        pb = make_progress(steps)
        for i in range(steps):
            ...
            pb.update(1)
    """
    import comfy.utils
    return comfy.utils.ProgressBar(total)
