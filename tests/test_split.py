# tests/test_split.py
import sys
import types
import numpy as np
import pytest


def _ensure_comfy_mock():
    """Ensure comfy.model_management is in sys.modules as a mock."""
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


_ensure_comfy_mock()


def test_split_produces_multiple_parts():
    from core.split import split_mesh_by_labels
    import trimesh

    box = trimesh.creation.box()
    # Create labels: first half faces = label 0, second half = label 1
    half = len(box.faces) // 2
    labels = np.array([0] * half + [1] * (len(box.faces) - half), dtype=np.int32)

    parts = split_mesh_by_labels(box, labels, min_faces=1)
    assert len(parts) == 2
    for p in parts:
        assert isinstance(p, (trimesh.Trimesh, trimesh.Scene))


def test_split_filters_small_parts():
    from core.split import split_mesh_by_labels
    import trimesh

    box = trimesh.creation.box()
    labels = np.zeros(len(box.faces), dtype=np.int32)
    labels[0] = 1  # Only 1 face in label 1

    parts = split_mesh_by_labels(box, labels, min_faces=10)
    assert len(parts) == 1  # label 1 filtered out (only 1 face)


def test_split_empty_labels_raises():
    from core.split import split_mesh_by_labels
    import trimesh

    box = trimesh.creation.box()
    labels = np.full(len(box.faces), -1, dtype=np.int32)
    with pytest.raises(ValueError, match="No valid segments"):
        split_mesh_by_labels(box, labels, min_faces=1)
