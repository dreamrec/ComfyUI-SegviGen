"""
Verify all SegviGen nodes register with correct INPUT_TYPES, RETURN_TYPES,
FUNCTION, and CATEGORY — without loading any GPU models.
"""
import sys
import types
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


def test_glb_to_voxel_registration():
    from nodes.nodes_voxel import SegviGenGLBtoVoxel
    node = SegviGenGLBtoVoxel
    assert node.CATEGORY == "SegviGen"
    assert node.FUNCTION == "convert"
    assert "SEGVIGEN_VOXEL" in node.RETURN_TYPES
    types_ = node.INPUT_TYPES()
    # Both inputs optional — at least one of trimesh/glb_path must be provided at runtime
    assert "trimesh" in types_.get("optional", {}) or "trimesh" in types_.get("required", {})
    assert "glb_path" in types_.get("optional", {}) or "glb_path" in types_.get("required", {})


def test_voxel_encode_registration():
    from nodes.nodes_voxel import SegviGenVoxelEncode
    node = SegviGenVoxelEncode
    assert node.CATEGORY == "SegviGen"
    assert "SEGVIGEN_SLAT" in node.RETURN_TYPES
    types_ = node.INPUT_TYPES()["required"]
    assert "model_config" in types_
    assert "voxel" in types_


def test_voxel_encode_output_shape(monkeypatch):
    """
    SegviGenVoxelEncode.encode() must return ({"latent": ..., "voxel": ...},)
    without calling any real GPU models. Use monkeypatch to stub the encoder.
    """
    import numpy as np
    from nodes.nodes_voxel import SegviGenVoxelEncode
    import core.pipeline  # must import first so monkeypatch can patch it

    dummy_voxel = {"grid": np.ones((64, 64, 64), dtype=bool), "metadata": {}}
    dummy_latent = object()  # opaque placeholder
    dummy_patcher = type("P", (), {"model": None})()

    monkeypatch.setattr(core.pipeline, "get_encoder_patcher", lambda cfg, d: dummy_patcher)
    monkeypatch.setattr(core.pipeline, "encode_voxel_to_slat", lambda m, g, c: dummy_latent)

    import comfy.model_management as mm_mod
    # Ensure these attributes exist on the mock (they may not be present on minimal mocks)
    if not hasattr(mm_mod, "load_models_gpu"):
        mm_mod.load_models_gpu = lambda _: None
    if not hasattr(mm_mod, "soft_empty_cache"):
        mm_mod.soft_empty_cache = lambda: None
    monkeypatch.setattr(mm_mod, "load_models_gpu", lambda _: None)
    monkeypatch.setattr(mm_mod, "soft_empty_cache", lambda: None)

    node = SegviGenVoxelEncode()
    result = node.encode(model_config={"resolution": "512"}, voxel=dummy_voxel)
    slat = result[0]
    assert "latent" in slat
    assert "voxel" in slat
    assert slat["latent"] is dummy_latent
    assert slat["voxel"] is dummy_voxel
