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


def test_preprocess_registration():
    from nodes.nodes_preprocess import SegviGenPreprocess
    node = SegviGenPreprocess
    assert node.CATEGORY == "SegviGen"
    assert "IMAGE" in node.RETURN_TYPES
    assert "MASK" in node.RETURN_TYPES
    types_ = node.INPUT_TYPES()
    assert "image" in types_["required"]
    assert "background_color" in types_.get("optional", {})


def test_conditioning_registration():
    from nodes.nodes_conditioning import SegviGenGetConditioning
    node = SegviGenGetConditioning
    assert node.CATEGORY == "SegviGen"
    assert "SEGVIGEN_COND" in node.RETURN_TYPES
    types_ = node.INPUT_TYPES()["required"]
    assert "model_config" in types_
    assert "image" in types_
    assert "mask" in types_


def test_full_sampler_registration():
    from nodes.nodes_sampler import SegviGenFullSampler
    node = SegviGenFullSampler
    assert node.CATEGORY == "SegviGen"
    assert "SEGVIGEN_SEG_RESULT" in node.RETURN_TYPES
    req = node.INPUT_TYPES()["required"]
    assert "model_config" in req
    assert "slat" in req
    assert "conditioning" in req
    opt = node.INPUT_TYPES().get("optional", {})
    assert "guidance_rescale" in opt
    assert "seed" in opt


def test_interactive_sampler_registration():
    from nodes.nodes_sampler import SegviGenInteractiveSampler
    node = SegviGenInteractiveSampler
    assert node.CATEGORY == "SegviGen"
    assert "SEGVIGEN_SEG_RESULT" in node.RETURN_TYPES
    req = node.INPUT_TYPES()["required"]
    assert "points" in req
    assert "slat" in req


def test_point_input_registration():
    from nodes.nodes_points import SegviGenPointInput
    node = SegviGenPointInput
    assert node.CATEGORY == "SegviGen"
    assert "SEGVIGEN_POINTS" in node.RETURN_TYPES
    req = node.INPUT_TYPES()["required"]
    assert "num_points" in req
    opt = node.INPUT_TYPES().get("optional", {})
    assert "point_1_x" in opt
    assert "point_10_z" in opt  # all 10 points present


def test_point_input_builds_correct_list():
    from nodes.nodes_points import SegviGenPointInput
    node = SegviGenPointInput()
    result, = node.build_points(num_points=2, point_1_x=10, point_1_y=20, point_1_z=30,
                                 point_2_x=5, point_2_y=5, point_2_z=5)
    assert result == [[10, 20, 30], [5, 5, 5]]


def test_point_input_respects_num_points():
    from nodes.nodes_points import SegviGenPointInput
    node = SegviGenPointInput()
    result, = node.build_points(num_points=1, point_1_x=7, point_1_y=8, point_1_z=9,
                                 point_2_x=1, point_2_y=2, point_2_z=3)
    assert len(result) == 1  # only 1 point despite point_2 being provided
