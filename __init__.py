import importlib.util as _ilu
import logging
import os
import sys
import types as _types

log = logging.getLogger("segvigen")

# ---------------------------------------------------------------------------
# comfy_aimdo stub
# ---------------------------------------------------------------------------
# ComfyUI Desktop added comfy_aimdo.host_buffer in a recent update.
# The TRELLIS2 comfy-env pixi worker may have an older comfy_aimdo that
# pre-dates this submodule.  Install a no-op stub so comfy.pinned_memory
# can still be imported inside the worker without crashing.
# setdefault() means we never replace a real, functioning installation.
try:
    import comfy_aimdo.host_buffer  # noqa: F401 — just test importability
except (ImportError, ModuleNotFoundError):
    class _StubModule(_types.ModuleType):
        """Returns a no-op callable for any unknown attribute."""
        def __getattr__(self, name):
            return lambda *a, **kw: None

    sys.modules.setdefault("comfy_aimdo", _StubModule("comfy_aimdo"))
    sys.modules.setdefault("comfy_aimdo.host_buffer",
                           _StubModule("comfy_aimdo.host_buffer"))

# Absolute path to this package directory.  Must be on sys.path so that
# absolute imports inside node sub-files (e.g. `from core.voxel import ...`)
# resolve to *our* core/ directory rather than something else on the path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def _find_trellis2_nodes():
    """
    Add ComfyUI-TRELLIS2/nodes/ to sys.path so 'trellis2', 'stages', 'rembg'
    all resolve to the ComfyUI-adapted copies (not the research-code root).

    ComfyUI-TRELLIS2 layout:
      ComfyUI-TRELLIS2/
        trellis2/          <- research code (missing samplers.py, vae.py etc.)
        nodes/
          trellis2/        <- ComfyUI-adapted copy  <- THIS is what we want
          stages.py        <- top-level conditioning/encoding stages
          rembg/           <- BiRefNet wrapper

    No try/except: if folder_paths is unavailable the package cannot function.
    Tests mock folder_paths in conftest.py before importing this module.
    """
    import folder_paths
    # folder_paths.base_path IS the ComfyUI root directory
    custom_nodes = os.path.join(folder_paths.base_path, "custom_nodes")

    trellis2_nodes = os.path.join(custom_nodes, "ComfyUI-TRELLIS2", "nodes")
    if not os.path.isdir(trellis2_nodes):
        raise ImportError(
            "ComfyUI-SegviGen requires ComfyUI-TRELLIS2 to be installed. "
            "Please install it via ComfyUI Manager first."
        )
    if trellis2_nodes not in sys.path:
        sys.path.insert(0, trellis2_nodes)
        log.info(f"SegviGen: added TRELLIS2 nodes path: {trellis2_nodes}")


def _load_nodes_package():
    """
    Load our nodes/ sub-package by file path, bypassing sys.modules['nodes'].

    ComfyUI's own nodes.py is loaded early and cached as sys.modules['nodes'],
    so a plain `from nodes import ...` would silently return ComfyUI's mappings
    instead of ours.  Using importlib with a unique module name avoids that.
    """
    _nodes_dir = os.path.join(_THIS_DIR, "nodes")
    _nodes_init = os.path.join(_nodes_dir, "__init__.py")
    _mod_name = "ComfyUI_SegviGen_nodes"

    spec = _ilu.spec_from_file_location(
        _mod_name,
        _nodes_init,
        submodule_search_locations=[_nodes_dir],
    )
    mod = _ilu.module_from_spec(spec)
    # Register before exec so intra-package relative imports resolve correctly.
    sys.modules[_mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_find_trellis2_nodes()
_nodes_mod = _load_nodes_package()

NODE_CLASS_MAPPINGS = _nodes_mod.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = _nodes_mod.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
