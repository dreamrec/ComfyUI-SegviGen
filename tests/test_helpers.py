import torch
import numpy as np
from PIL import Image
import pytest


def test_tensor_to_pil_rgb():
    from nodes.helpers import tensor_to_pil
    t = torch.rand(1, 64, 64, 3)
    img = tensor_to_pil(t)
    assert isinstance(img, Image.Image)
    assert img.size == (64, 64)
    assert img.mode == "RGB"


def test_tensor_to_pil_strips_batch():
    from nodes.helpers import tensor_to_pil
    # Both [1,H,W,C] and [H,W,C] should work
    t = torch.rand(64, 64, 3)
    img = tensor_to_pil(t)
    assert img.size == (64, 64)


def test_pil_to_tensor_shape():
    from nodes.helpers import pil_to_tensor
    img = Image.new("RGB", (32, 32), color=(128, 64, 32))
    t = pil_to_tensor(img)
    assert t.shape == (1, 32, 32, 3)
    assert t.dtype == torch.float32
    assert t.min() >= 0.0 and t.max() <= 1.0


def test_pil_to_tensor_rgba_converted():
    from nodes.helpers import pil_to_tensor
    img = Image.new("RGBA", (16, 16), color=(255, 0, 0, 128))
    t = pil_to_tensor(img)
    assert t.shape == (1, 16, 16, 3)  # RGBA stripped to RGB


def test_check_interrupt_no_raise(monkeypatch):
    """check_interrupt should not raise when processing continues."""
    import sys
    import types
    # Mock comfy.model_management before importing helpers
    mm = types.ModuleType("model_management")
    mm.throw_exception_if_processing_interrupted = lambda: None
    comfy_module = types.ModuleType("comfy")
    comfy_module.model_management = mm
    sys.modules["comfy"] = comfy_module
    sys.modules["comfy.model_management"] = mm
    
    import nodes.helpers as h
    h.check_interrupt()  # must not raise


def test_make_progress_returns_callable():
    import sys
    import types
    # Mock comfy.utils before importing helpers
    progress_class = type("ProgressBar", (), {"__init__": lambda self, total: None, "update": lambda self, n: None})
    utils_module = types.ModuleType("utils")
    utils_module.ProgressBar = progress_class
    comfy_module = sys.modules.get("comfy") or types.ModuleType("comfy")
    comfy_module.utils = utils_module
    sys.modules["comfy"] = comfy_module
    sys.modules["comfy.utils"] = utils_module
    
    from nodes.helpers import make_progress
    # Should return an object with .update(n) method
    pb = make_progress(10)
    assert hasattr(pb, "update")
