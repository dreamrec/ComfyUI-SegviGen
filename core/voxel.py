"""
GLB mesh -> voxel grid conversion for SegviGen.

Takes either a trimesh.Trimesh/Scene object or a path to a .glb file.
Returns a SEGVIGEN_VOXEL dict:
  {
    "grid": np.ndarray bool [R, R, R],
    "metadata": {
        "transform": np.ndarray [4,4] (voxel->world),
        "original_bounds": (np.ndarray [3], np.ndarray [3])  # (min, max)
    }
  }
"""
import logging
import numpy as np

log = logging.getLogger("segvigen")


def _load_mesh(source) -> "trimesh.Trimesh":
    """Load a trimesh from a path string or return the passed Trimesh/Scene."""
    import trimesh
    if isinstance(source, (str, bytes)):
        loaded = trimesh.load(str(source), force="mesh")
    elif isinstance(source, trimesh.Scene):
        loaded = source.dump(concatenate=True)
    else:
        loaded = source

    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Could not load mesh from {source!r} — got {type(loaded)}")
    return loaded


def _normalize_to_unit_cube(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """Translate and scale mesh so it fits inside [-0.5, 0.5]^3."""
    mesh = mesh.copy()
    bounds_min = mesh.bounds[0]
    bounds_max = mesh.bounds[1]
    center = (bounds_min + bounds_max) / 2.0
    extent = (bounds_max - bounds_min).max()
    if extent < 1e-8:
        raise ValueError("Mesh has zero extent — cannot normalize")
    mesh.apply_translation(-center)
    mesh.apply_scale(1.0 / extent)
    return mesh


def _repair_mesh(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """Best-effort watertight repair using trimesh utilities."""
    import trimesh
    mesh = mesh.copy()
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh)
    return mesh


def mesh_to_voxel_grid(source, resolution: int = 64, simplify_faces: int = 100_000) -> dict:
    """
    Convert a mesh to a boolean voxel grid.

    Args:
        source: trimesh.Trimesh, trimesh.Scene, or path string to .glb/.obj etc.
        resolution: voxel grid side length (32, 64, or 128)
        simplify_faces: target face count before voxelization (reduces memory)

    Returns:
        SEGVIGEN_VOXEL dict with 'grid' and 'metadata' keys.

    Raises:
        ValueError: if mesh cannot be loaded or produces an empty voxel grid.
    """
    import trimesh

    mesh = _load_mesh(source)

    # An empty trimesh has bounds == None; treat as empty voxel grid immediately.
    if mesh.bounds is None or len(mesh.faces) == 0:
        raise ValueError(
            "Mesh produced empty voxel grid — check mesh is watertight and non-degenerate"
        )

    original_bounds = (mesh.bounds[0].copy(), mesh.bounds[1].copy())

    # Simplify if mesh is very dense
    if len(mesh.faces) > simplify_faces:
        log.info(f"Simplifying mesh from {len(mesh.faces)} to {simplify_faces} faces")
        mesh = mesh.simplify_quadric_decimation(simplify_faces)

    mesh = _repair_mesh(mesh)
    mesh = _normalize_to_unit_cube(mesh)

    # Voxelize: pitch = 1/resolution maps to [-0.5, 0.5] cube
    pitch = 1.0 / resolution
    voxel_grid = mesh.voxelized(pitch=pitch)
    voxel_grid = voxel_grid.fill()

    grid = voxel_grid.matrix  # bool ndarray

    if grid.sum() == 0:
        raise ValueError(
            "Mesh produced empty voxel grid — check mesh is watertight and non-degenerate"
        )

    # Crop or pad to exactly [resolution, resolution, resolution], centered.
    # Centering ensures the mesh occupies the middle of the voxel cube regardless
    # of whether the trimesh grid came out slightly smaller or larger than 'resolution'.
    target = np.zeros((resolution, resolution, resolution), dtype=bool)
    slices_grid = [slice(None)] * 3
    slices_target = [slice(None)] * 3
    for ax in range(3):
        gd = grid.shape[ax]
        if gd >= resolution:
            start_g = (gd - resolution) // 2
            slices_grid[ax] = slice(start_g, start_g + resolution)
            slices_target[ax] = slice(0, resolution)
        else:
            start_t = (resolution - gd) // 2
            slices_grid[ax] = slice(0, gd)
            slices_target[ax] = slice(start_t, start_t + gd)
    target[tuple(slices_target)] = grid[tuple(slices_grid)]

    return {
        "grid": target,
        "metadata": {
            "transform": voxel_grid.transform,
            "original_bounds": original_bounds,
        },
    }
