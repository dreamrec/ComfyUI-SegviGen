import logging
import os
import sys

log = logging.getLogger("segvigen")


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


_find_trellis2_nodes()

from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
