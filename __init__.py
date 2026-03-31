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

# Serve web/picker.html and web/mesh_picker_widget.js as ComfyUI frontend files.
WEB_DIRECTORY = "./web"

# ---------------------------------------------------------------------------
# Custom HTTP route: picker → server → WebSocket → JS extension
# ---------------------------------------------------------------------------
# picker.html POSTs confirmed points here; we relay them to all WS clients
# so the JS extension can update the node widget — no window.opener needed.
def _run_preview_inference(cache: dict, points_list: list, steps: int = 2) -> dict:
    """
    Geometric connected-component preview for the 3D picker.

    Uses a KD-tree radius-based BFS rather than 6-connected grid flood-fill.
    The 512-res SLAT typically has only ~1 000–2 000 occupied voxels spread over
    a 512³ grid (average NN distance ≈ 7 voxel units), so 6-connectivity finds
    components of size 1.  Instead we:
      1. Build a scipy cKDTree over all occupied voxel coords.
      2. Compute an adaptive radius = 2.5 × median nearest-neighbour distance
         (floor at 2, cap at 20) so the BFS jumps gaps in sparse point clouds.
      3. BFS by expanding each frontier voxel with query_ball_point(radius).

    Returns a dict ready to JSON-serialise: {ok, components, voxel_resolution}.
    """
    import numpy as np
    from collections import deque
    from scipy.spatial import cKDTree

    coords_np        = cache["coords_np"]
    voxel_resolution = cache["voxel_resolution"]
    N = len(coords_np)

    log.info(
        f"SegviGen preview BFS: {N} occupied voxels, "
        f"R={voxel_resolution}, {len(points_list)} click points"
    )

    if N == 0:
        log.warning("SegviGen preview BFS: ZERO voxels — occupancy grid may be empty")
        return {"ok": True, "components": [], "voxel_resolution": voxel_resolution}

    # Build KD-tree (float64 coords for accurate distance queries)
    tree = cKDTree(coords_np.astype(np.float64))

    # ── Adaptive radius: 2.5× median nearest-neighbour distance ─────────────
    # Sample up to 200 points to estimate the NN distribution quickly.
    sample_n = min(200, N)
    sample_idx = np.random.choice(N, sample_n, replace=False)
    nn_dists, _ = tree.query(coords_np[sample_idx].astype(np.float64), k=2)
    median_nn = float(np.median(nn_dists[:, 1]))          # col-1: nearest neighbour
    radius = max(2.0, min(20.0, median_nn * 2.5))
    log.info(f"SegviGen preview BFS: adaptive radius={radius:.1f} (median NN={median_nn:.1f})")

    # Index array → tuple for fast membership lookups
    coord_tuples = [tuple(c) for c in coords_np.tolist()]
    tuple_to_idx  = {t: i for i, t in enumerate(coord_tuples)}

    labeled: set = set()    # indices already assigned
    components: list = []

    for pt in points_list[:10]:
        px = max(0, min(voxel_resolution - 1, int(round(pt[0]))))
        py = max(0, min(voxel_resolution - 1, int(round(pt[1]))))
        pz = max(0, min(voxel_resolution - 1, int(round(pt[2]))))

        # Snap to nearest occupied voxel
        click_pt = np.array([[px, py, pz]], dtype=np.float64)
        _, nn_idx = tree.query(click_pt, k=1)
        start_idx = int(nn_idx[0])
        snap = coord_tuples[start_idx]
        if (px, py, pz) != snap:
            log.info(f"SegviGen preview BFS: click ({px},{py},{pz}) snapped to {snap}")

        if start_idx in labeled:
            continue

        # ── KD-tree radius BFS ────────────────────────────────────────────────
        visited_idx: set = {start_idx}
        queue: deque = deque([start_idx])
        component: list = []

        while queue:
            curr_idx = queue.popleft()
            if curr_idx in labeled:
                continue
            component.append(list(coord_tuples[curr_idx]))
            labeled.add(curr_idx)

            nb_indices = tree.query_ball_point(
                coords_np[curr_idx].astype(np.float64), radius
            )
            for nb_idx in nb_indices:
                if nb_idx not in visited_idx:
                    visited_idx.add(nb_idx)
                    queue.append(nb_idx)

        components.append(component)
        log.info(
            f"SegviGen preview BFS: click ({px},{py},{pz}) → "
            f"component {len(components)} with {len(component)} voxels"
        )

    total = sum(len(c) for c in components)
    if total:
        log.info(f"SegviGen preview BFS: {len(components)} components, {total} voxels total")
    else:
        log.warning("SegviGen preview BFS: ZERO voxels in any component")

    return {"ok": True, "components": components, "voxel_resolution": voxel_resolution}


try:
    from aiohttp import web as _web
    from server import PromptServer as _PS

    @_PS.instance.routes.post("/segvigen/confirm_points")
    async def _segvigen_confirm_points(request):
        data   = await request.json()
        node_id = str(data.get("nodeId", ""))
        points  = data.get("points", [])
        # Broadcast to all connected ComfyUI browser clients via WebSocket.
        await _PS.instance.send_json(
            "segvigen.points_confirmed",
            {"nodeId": node_id, "points": points},
        )
        return _web.json_response({"ok": True, "count": len(points)})

    @_PS.instance.routes.post("/segvigen/preview_segment")
    async def _segvigen_preview_segment(request):
        """
        Geometric BFS connected-component preview for the 3D picker.

        Body: { nodeId: str, points: [[x,y,z], ...] }
        Returns: { ok: true, voxels: [[x,y,z], ...], voxel_resolution: int }
                 { ok: false, error: str }
        """
        import asyncio
        data    = await request.json()
        node_id = str(data.get("nodeId", ""))
        points  = data.get("points", [])
        steps   = int(data.get("steps", 2))

        from core.preview_cache import retrieve as _cache_get
        cache = _cache_get(node_id)
        if cache is None:
            return _web.json_response({
                "ok": False,
                "error": "no_cache",
                "message": "Run the workflow once to enable live preview.",
            })

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None, _run_preview_inference, cache, points, steps
            )
        except Exception as exc:
            import traceback as _tb
            log.warning(f"SegviGen preview_segment error: {exc}\n{_tb.format_exc()}")
            return _web.json_response({"ok": False, "error": str(exc)})

        return _web.json_response(result)

    log.info("SegviGen: registered /segvigen/confirm_points and /segvigen/preview_segment routes")
except Exception as _e:
    log.warning(f"SegviGen: could not register confirm_points route: {_e}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
