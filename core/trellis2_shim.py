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
    ensure_supported_trellis2_revision(stages_mod)
    return stages_mod


# ── Required TRELLIS2 private API surface ─────────────────────────────────────
# These are the private functions SegviGen depends on. If any are missing or
# have incompatible signatures, the installed TRELLIS2 is too old/new.
_REQUIRED_CALLABLES = [
    "_init_config",
    "_sample_tex_slat",
    "_decode_tex_slat",
]

_compat_checked = False


def ensure_supported_trellis2_revision(stages_mod=None):
    """
    Validate that the loaded TRELLIS2 stages module exposes the private
    functions SegviGen depends on. Called once on first load.

    Raises ImportError with an actionable message if incompatible.
    """
    global _compat_checked
    if _compat_checked:
        return

    if stages_mod is None:
        stages_key = f"{_SHIM_PKG}.stages"
        stages_mod = sys.modules.get(stages_key)
        if stages_mod is None:
            return  # not loaded yet, will check on first load

    missing = []
    for name in _REQUIRED_CALLABLES:
        fn = getattr(stages_mod, name, None)
        if fn is None or not callable(fn):
            missing.append(name)

    if missing:
        raise ImportError(
            f"SegviGen: installed ComfyUI-TRELLIS2 is incompatible — "
            f"missing required functions in stages.py: {', '.join(missing)}.\n"
            f"Please update ComfyUI-TRELLIS2 to a version that includes "
            f"_init_config, _sample_tex_slat, and _decode_tex_slat."
        )

    _compat_checked = True
    log.info("SegviGen: TRELLIS2 compatibility check passed")
