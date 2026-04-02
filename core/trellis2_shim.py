"""
Load TRELLIS2's nodes/stages.py via a synthetic package shim.

Extracted here so both core/ and nodes/ modules can import it without
hitting the `nodes` namespace collision with ComfyUI's own nodes module.
"""
import importlib.machinery
import importlib.util
import logging
import os
import sys
import types

log = logging.getLogger("segvigen")

# Synthetic package name — never conflicts with a real package.
_SHIM_PKG = "segvigen_trellis2_stages_shim"


def load_trellis2_stages():
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
    spec.loader.exec_module(stages_mod)

    log.info("SegviGen: loaded TRELLIS2 nodes/stages.py via package shim")
    return stages_mod
