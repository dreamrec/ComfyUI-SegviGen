"""Shared test fixtures for ComfyUI-SegviGen tests."""
import sys
import os
import pytest

# Make package importable without ComfyUI present
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Add ComfyUI to sys.path so comfy modules can be imported
# conftest is in tests/, go up to ComfyUI-SegviGen, then to custom_nodes, then to ComfyUI
_COMFYUI_ROOT_FOR_IMPORTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if os.path.isdir(_COMFYUI_ROOT_FOR_IMPORTS) and os.path.exists(os.path.join(_COMFYUI_ROOT_FOR_IMPORTS, "comfy")):
    sys.path.insert(0, _COMFYUI_ROOT_FOR_IMPORTS)

# Mock folder_paths BEFORE importing the package.
# Use unconditional assignment (not setdefault) so the mock always wins,
# even in environments where a partial ComfyUI install might provide folder_paths.
# __file__ = tests/conftest.py -> 3 levels up = ComfyUI root
import types
_fp = types.ModuleType("folder_paths")
_COMFYUI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_fp.base_path = _COMFYUI_ROOT
_fp.models_dir = os.path.join(_COMFYUI_ROOT, "models")
_fp.output_directory = os.path.join(_COMFYUI_ROOT, "output")
_fp.folder_names_and_paths = {}

def _get_folder_paths(folder_name):
    return []

_fp.get_folder_paths = _get_folder_paths
sys.modules["folder_paths"] = _fp  # unconditional -- always use the mock in tests


@pytest.fixture
def cube_trimesh():
    """A simple watertight cube mesh for voxel tests."""
    import trimesh
    return trimesh.creation.box(extents=[1.0, 1.0, 1.0])


@pytest.fixture
def dummy_image_tensor():
    """A 1x64x64x3 float32 ComfyUI IMAGE tensor."""
    import torch
    return torch.rand(1, 64, 64, 3, dtype=torch.float32)


@pytest.fixture
def dummy_mask_tensor():
    """A 1x64x64 float32 ComfyUI MASK tensor (all foreground)."""
    import torch
    return torch.ones(1, 64, 64, dtype=torch.float32)
