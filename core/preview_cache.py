"""
File-based preview cache for SegviGen real-time picker preview.

The sampler nodes run in comfy-env isolation workers (separate processes),
while the HTTP route handler runs in the main ComfyUI process.  An in-memory
dict cannot cross this boundary, so we serialise the minimal data needed for
BFS preview (voxel coords + resolution) to a temp .npz file on disk.

Keyed by node_id (str).  A "latest" symlink always points to the most recent
cache so the picker (which may use a different LiteGraph node ID) can find it.
"""
import logging
import os
import tempfile

import numpy as np

log = logging.getLogger("segvigen")

_CACHE_DIR = os.path.join(tempfile.gettempdir(), "segvigen_preview_cache")
_LATEST_FILE = os.path.join(_CACHE_DIR, "_latest_node_id.txt")


def _ensure_dir():
    os.makedirs(_CACHE_DIR, exist_ok=True)


def _path_for(node_id: str) -> str:
    return os.path.join(_CACHE_DIR, f"node_{node_id}.npz")


def store(node_id: str, state: dict) -> None:
    """Persist the SLAT coords + voxel_resolution to disk (called from worker)."""
    import torch

    _ensure_dir()
    node_id = str(node_id)

    slat = state["slat"]
    voxel_resolution = int(state["voxel_resolution"])

    raw_coords = slat["latent"].coords
    if isinstance(raw_coords, torch.Tensor):
        coords_np = raw_coords.detach().cpu().numpy().astype(np.int32)
    else:
        coords_np = np.asarray(raw_coords, dtype=np.int32)

    # Drop batch column: [N,4] → [N,3]
    if coords_np.ndim == 2 and coords_np.shape[1] == 4:
        coords_np = coords_np[:, 1:]

    path = _path_for(node_id)
    np.savez_compressed(path, coords=coords_np, voxel_resolution=np.array([voxel_resolution]))
    log.info(f"SegviGen preview cache: saved {coords_np.shape[0]} coords to {path}")

    # Mark as latest
    try:
        with open(_LATEST_FILE, "w") as f:
            f.write(node_id)
    except OSError:
        pass


def retrieve(node_id: str) -> dict | None:
    """Load cached coords from disk (called from main process HTTP route)."""
    _ensure_dir()
    node_id = str(node_id)
    path = _path_for(node_id)

    # Try exact node_id first, then fall back to latest
    if not os.path.exists(path):
        try:
            with open(_LATEST_FILE, "r") as f:
                latest_id = f.read().strip()
            path = _path_for(latest_id)
        except (OSError, FileNotFoundError):
            return None

    if not os.path.exists(path):
        return None

    try:
        data = np.load(path)
        coords_np = data["coords"]
        voxel_resolution = int(data["voxel_resolution"][0])
        log.info(
            f"SegviGen preview cache: loaded {coords_np.shape[0]} coords, "
            f"R={voxel_resolution} from {path}"
        )
        return {
            "coords_np": coords_np,
            "voxel_resolution": voxel_resolution,
        }
    except Exception as exc:
        log.warning(f"SegviGen preview cache: failed to load {path}: {exc}")
        return None


def clear(node_id: str) -> None:
    path = _path_for(str(node_id))
    try:
        os.remove(path)
    except OSError:
        pass
