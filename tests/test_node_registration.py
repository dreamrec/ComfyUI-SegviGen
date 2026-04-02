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
    assert "glb_file" in types_.get("optional", {}) or "glb_file" in types_.get("required", {})


def test_voxel_encode_registration():
    from nodes.nodes_voxel import SegviGenVoxelEncode
    node = SegviGenVoxelEncode
    assert node.CATEGORY == "SegviGen"
    assert "SEGVIGEN_SLAT" in node.RETURN_TYPES
    types_ = node.INPUT_TYPES()["required"]
    assert "model_config" in types_
    assert "shape_result" in types_
    assert "conditioning" in types_


def test_voxel_encode_output_shape():
    """
    Verify SegviGenVoxelEncode registration — the actual encode() requires
    TRELLIS2 stages, so we only test the class-level contract here.
    """
    from nodes.nodes_voxel import SegviGenVoxelEncode
    node = SegviGenVoxelEncode
    assert node.CATEGORY == "SegviGen"
    assert node.FUNCTION == "encode"
    assert node.RETURN_TYPES == ("SEGVIGEN_SLAT",)
    types_ = node.INPUT_TYPES()
    assert "model_config" in types_["required"]
    assert "shape_result" in types_["required"]
    assert "conditioning" in types_["required"]
    opt = types_.get("optional", {})
    assert "seed" in opt
    assert "tex_guidance_strength" in opt
    assert "tex_sampling_steps" in opt


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


def test_render_preview_registration():
    from nodes.nodes_output import SegviGenRenderPreview
    node = SegviGenRenderPreview
    assert node.CATEGORY == "SegviGen"
    assert "IMAGE" in node.RETURN_TYPES
    req = node.INPUT_TYPES()["required"]
    assert "seg_result" in req


def test_export_parts_registration():
    from nodes.nodes_output import SegviGenExportParts
    node = SegviGenExportParts
    assert node.CATEGORY == "SegviGen"
    assert "STRING" in node.RETURN_TYPES
    assert node.OUTPUT_NODE is True
    opt = node.INPUT_TYPES().get("optional", {})
    assert "max_faces" in opt
    assert "min_segment_faces" in opt


def test_all_nodes_in_mappings():
    from nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    expected = {
        "SegviGenLoadMesh", "SegviGenGLBtoVoxel", "SegviGenVoxelEncode",
        "SegviGenFromShapeResult", "SegviGenPreprocess",
        "SegviGenGetConditioning", "SegviGenNullConditioning",
        "SegviGenFullSampler", "SegviGenInteractiveSampler",
        "SegviGenPointInput", "SegviGenMeshPicker",
        "SegviGenRenderPreview", "SegviGenExportParts",
        "SegviGenAssetPrepare", "SegviGenAssetEncode",
        "SegviGenGet2DMapConditioning",
    }
    assert set(NODE_CLASS_MAPPINGS.keys()) == expected
    assert set(NODE_DISPLAY_NAME_MAPPINGS.keys()) == expected
