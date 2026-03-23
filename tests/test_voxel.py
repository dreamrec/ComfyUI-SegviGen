# tests/test_voxel.py
import numpy as np
import pytest


def test_glb_to_voxel_cube(cube_trimesh):
    """A unit cube should produce a non-empty voxel grid."""
    from core.voxel import mesh_to_voxel_grid
    result = mesh_to_voxel_grid(cube_trimesh, resolution=32)
    assert "grid" in result
    assert "metadata" in result
    grid = result["grid"]
    assert grid.shape == (32, 32, 32)
    assert grid.dtype == bool
    assert grid.sum() > 0  # at least some voxels filled


def test_glb_to_voxel_metadata_has_transform(cube_trimesh):
    from core.voxel import mesh_to_voxel_grid
    result = mesh_to_voxel_grid(cube_trimesh, resolution=32)
    meta = result["metadata"]
    assert "transform" in meta
    assert "original_bounds" in meta


def test_empty_mesh_raises():
    import trimesh
    from core.voxel import mesh_to_voxel_grid
    empty = trimesh.Trimesh()
    with pytest.raises(ValueError, match="empty voxel grid"):
        mesh_to_voxel_grid(empty, resolution=32)


def test_resolution_respected(cube_trimesh):
    from core.voxel import mesh_to_voxel_grid
    for res in (32, 64):
        result = mesh_to_voxel_grid(cube_trimesh, resolution=res)
        assert result["grid"].shape == (res, res, res)


def test_load_glb_from_path(tmp_path, cube_trimesh):
    """Load from a .glb file path rather than a trimesh object."""
    from core.voxel import mesh_to_voxel_grid
    import trimesh
    glb_path = str(tmp_path / "cube.glb")
    cube_trimesh.export(glb_path)
    result = mesh_to_voxel_grid(glb_path, resolution=32)
    assert result["grid"].sum() > 0
