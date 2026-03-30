"""
SegviGen conditioning nodes:
  - SegviGenGetConditioning: DINOv3 visual features from an image+mask
  - SegviGenNullConditioning: null embedding for unconditioned auto-segmentation

Import strategy for run_conditioning:
  TRELLIS2 has two stages.py files:
    nodes/stages.py           — main process, returns tensors directly  ← we want this
    nodes/trellis_utils/stages.py — subprocess IPC, returns disk-file refs

  nodes/stages.py uses `from .helpers import smart_crop_square` (relative import).
  Importing it as a top-level module (via sys.path) fails because there is no
  parent package to resolve `.helpers` against.

  Fix: _load_trellis2_stages() registers a synthetic parent package in sys.modules
  before exec-ing stages.py, so `from .helpers import ...` resolves correctly.

Returns SEGVIGEN_COND dict with keys:
  {
    "cond_512":  torch.Tensor,
    "neg_cond":  torch.Tensor,
    "cond_1024": torch.Tensor,  # present when resolution in 1024_cascade/1536_cascade/1024
  }
"""
import logging
import os
import sys
import types
import importlib.util
import importlib.machinery

from .helpers import check_interrupt

log = logging.getLogger("segvigen")

# Synthetic package name — never conflicts with a real package.
_SHIM_PKG = "segvigen_trellis2_stages_shim"


def _load_trellis2_stages():
    """
    Load TRELLIS2's nodes/stages.py with a proper package context so its
    relative imports (from .helpers, from .trellis2) resolve correctly.

    Registers a synthetic parent package '_SHIM_PKG' in sys.modules whose
    __path__ points to the TRELLIS2 nodes/ directory.  Once registered,
    `from .helpers import smart_crop_square` inside stages.py becomes a
    lookup of `segvigen_trellis2_stages_shim.helpers`, which we also load.
    """
    stages_key = f"{_SHIM_PKG}.stages"
    if stages_key in sys.modules:
        return sys.modules[stages_key]

    # Locate the TRELLIS2 nodes/ directory on sys.path.
    nodes_dir = None
    for p in sys.path:
        if (p and
                os.path.isfile(os.path.join(p, "stages.py")) and
                os.path.isfile(os.path.join(p, "helpers.py")) and
                os.path.isdir(os.path.join(p, "trellis2"))):
            nodes_dir = p
            break
    if nodes_dir is None:
        raise ImportError(
            "SegviGen: TRELLIS2 nodes/ directory not found on sys.path.\n"
            "Ensure ComfyUI-TRELLIS2 is installed."
        )

    # 1. Register the synthetic parent package.
    if _SHIM_PKG not in sys.modules:
        pkg = types.ModuleType(_SHIM_PKG)
        pkg.__path__ = [nodes_dir]
        pkg.__package__ = _SHIM_PKG
        pkg.__spec__ = importlib.machinery.ModuleSpec(
            _SHIM_PKG, None, origin=nodes_dir, is_package=True
        )
        sys.modules[_SHIM_PKG] = pkg

    # 2. Load helpers.py as <shim>.helpers so `from .helpers import ...` resolves.
    helpers_key = f"{_SHIM_PKG}.helpers"
    if helpers_key not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            helpers_key, os.path.join(nodes_dir, "helpers.py")
        )
        helpers_mod = importlib.util.module_from_spec(spec)
        helpers_mod.__package__ = _SHIM_PKG
        sys.modules[helpers_key] = helpers_mod
        spec.loader.exec_module(helpers_mod)

    # 3. Load stages.py as <shim>.stages.
    spec = importlib.util.spec_from_file_location(
        stages_key, os.path.join(nodes_dir, "stages.py")
    )
    stages_mod = importlib.util.module_from_spec(spec)
    stages_mod.__package__ = _SHIM_PKG
    sys.modules[stages_key] = stages_mod
    # exec_module triggers `from .helpers import smart_crop_square` — now it
    # resolves to sys.modules['segvigen_trellis2_stages_shim.helpers'].
    spec.loader.exec_module(stages_mod)

    log.info("SegviGen: loaded TRELLIS2 nodes/stages.py via package shim")
    return stages_mod


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
        import comfy.model_management as mm

        check_interrupt()

        resolution = model_config.get("resolution", "1024_cascade")
        include_1024 = resolution in ("1024_cascade", "1536_cascade", "1024")

        log.info(f"SegviGen: extracting DinoV3 conditioning (resolution={resolution})")

        stages = _load_trellis2_stages()
        cond, _ = stages.run_conditioning(
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

        check_interrupt()

        resolution = model_config.get("resolution", "1024_cascade")
        include_1024 = resolution in ("1024_cascade", "1536_cascade", "1024")

        # Blank 512×512 black image — standard conditioning resolution.
        # ComfyUI IMAGE tensors are [B, H, W, C] float32 in [0, 1].
        blank_image = torch.zeros(1, 512, 512, 3)
        blank_mask = torch.ones(1, 512, 512)   # full white mask (no cropping)

        log.info("SegviGen: building null conditioning (blank image → DINOv3 null embedding)")

        stages = _load_trellis2_stages()
        cond_dict, _ = stages.run_conditioning(
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
