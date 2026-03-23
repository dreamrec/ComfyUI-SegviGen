"""
Integration smoke tests — verify the package loads and all nodes
register correctly in a ComfyUI-like environment (no GPU required).

These tests verify node graph plumbing but do NOT run inference.
"""
import sys
import types
import pytest

# Ensure comfy mock is present before any imports that need it
if "comfy" not in sys.modules:
    comfy_mod = types.ModuleType("comfy")
    sys.modules["comfy"] = comfy_mod
if "comfy.model_management" not in sys.modules:
    mm_mod = types.ModuleType("comfy.model_management")
    mm_mod.load_models_gpu = lambda models: None
    mm_mod.soft_empty_cache = lambda: None
    mm_mod.throw_exception_if_processing_interrupted = lambda: None
    sys.modules["comfy.model_management"] = mm_mod
    sys.modules["comfy"].model_management = mm_mod


def test_package_imports():
    """Package root __init__.py must export NODE_CLASS_MAPPINGS (TRELLIS2 path is mocked in conftest)."""
    import importlib.util
    import os

    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    spec = importlib.util.spec_from_file_location(
        "comfyui_segvigen",
        os.path.join(pkg_root, "__init__.py"),
        submodule_search_locations=[pkg_root],
    )
    m = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec_module so imports can resolve
    sys.modules["comfyui_segvigen"] = m
    spec.loader.exec_module(m)
    assert m.NODE_CLASS_MAPPINGS is not None
    assert len(m.NODE_CLASS_MAPPINGS) == 9


def test_type_strings_are_consistent():
    """All custom type strings used in inputs must appear somewhere in RETURN_TYPES."""
    from nodes import NODE_CLASS_MAPPINGS

    # Collect all output types
    all_outputs = set()
    for cls in NODE_CLASS_MAPPINGS.values():
        all_outputs.update(cls.RETURN_TYPES)

    # Collect all custom input types (non-ComfyUI-standard)
    standard_types = {"IMAGE", "MASK", "STRING", "INT", "FLOAT", "BOOLEAN",
                      "TRIMESH", "TRELLIS2_MODEL_CONFIG"}
    segvigen_input_types = set()
    for cls in NODE_CLASS_MAPPINGS.values():
        for category in cls.INPUT_TYPES().values():
            for _name, type_def in category.items():
                t = type_def[0] if isinstance(type_def, tuple) else type_def
                if isinstance(t, str) and t.startswith("SEGVIGEN_"):
                    segvigen_input_types.add(t)

    for t in segvigen_input_types:
        assert t in all_outputs, (
            f"Input type '{t}' is consumed but never produced by any node. "
            "Check that the producing node's RETURN_TYPES includes it."
        )


def test_voxel_roundtrip(cube_trimesh):
    """mesh_to_voxel_grid must return a non-empty grid for a unit cube."""
    from core.voxel import mesh_to_voxel_grid
    result = mesh_to_voxel_grid(cube_trimesh, resolution=32)
    assert result["grid"].sum() > 0


def test_split_roundtrip(cube_trimesh):
    """split_mesh_by_labels must return at least 1 part for a labeled mesh."""
    import numpy as np
    from core.split import split_mesh_by_labels
    labels = np.zeros(len(cube_trimesh.faces), dtype=np.int32)
    parts = split_mesh_by_labels(cube_trimesh, labels, min_faces=1)
    assert len(parts) >= 1
