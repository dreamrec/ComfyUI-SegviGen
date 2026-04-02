"""
Tests for core/contracts.py — payload constructors and validators.
"""
import pytest
import numpy as np


def test_build_segvigen_slat_minimal():
    from core.contracts import build_segvigen_slat, SEGVIGEN_CONTRACT_VERSION
    slat = build_segvigen_slat("mock_shape_slat")
    assert slat["segvigen_contract_version"] == SEGVIGEN_CONTRACT_VERSION
    assert slat["shape_slat"] == "mock_shape_slat"
    assert slat["latent"] == "mock_shape_slat"  # backward-compat alias
    assert slat["source"] == "shape_only"
    assert slat["tex_slat"] is None
    assert slat["subs"] is None
    assert slat["pipeline_type"] == "512"
    assert slat["normalization"] is None


def test_build_segvigen_slat_full():
    from core.contracts import build_segvigen_slat, SOURCE_BRIDGE_FULL
    slat = build_segvigen_slat(
        "shape", tex_slat="tex", subs=["sub1"],
        voxel_resolution=1024, source=SOURCE_BRIDGE_FULL,
        pipeline_type="1024",
    )
    assert slat["source"] == "bridge_full"
    assert slat["tex_slat"] == "tex"
    assert slat["subs"] == ["sub1"]
    assert slat["voxel"]["resolution"] == 1024


def test_build_segvigen_slat_invalid_source():
    from core.contracts import build_segvigen_slat
    with pytest.raises(ValueError, match="invalid source"):
        build_segvigen_slat("shape", source="bogus")


def test_validate_segvigen_slat_valid():
    from core.contracts import build_segvigen_slat, validate_segvigen_slat
    slat = build_segvigen_slat("shape")
    validate_segvigen_slat(slat)  # should not raise


def test_validate_segvigen_slat_require_tex_fails():
    from core.contracts import build_segvigen_slat, validate_segvigen_slat
    slat = build_segvigen_slat("shape")
    with pytest.raises(ValueError, match="requires tex_slat"):
        validate_segvigen_slat(slat, require_tex=True)


def test_validate_segvigen_slat_require_tex_passes():
    from core.contracts import build_segvigen_slat, validate_segvigen_slat, SOURCE_BRIDGE_FULL
    slat = build_segvigen_slat("shape", tex_slat="tex", source=SOURCE_BRIDGE_FULL)
    validate_segvigen_slat(slat, require_tex=True)  # should not raise


def test_get_shape_slat_new_key():
    from core.contracts import build_segvigen_slat, get_shape_slat
    slat = build_segvigen_slat("shape")
    assert get_shape_slat(slat) == "shape"


def test_get_shape_slat_legacy_key():
    from core.contracts import get_shape_slat
    # Simulate a legacy payload with only "latent" key
    legacy = {"latent": "old_shape"}
    assert get_shape_slat(legacy) == "old_shape"


def test_build_segvigen_seg_result():
    from core.contracts import (
        build_segvigen_seg_result, SEGVIGEN_CONTRACT_VERSION,
        MODE_INTERACTIVE_BINARY, LABELS_DECODED_BINARY,
    )
    labels = np.zeros((16, 16, 16), dtype=np.int32)
    result = build_segvigen_seg_result(
        output_tex_slat="mock_output",
        labels=labels,
        labels_source=LABELS_DECODED_BINARY,
        mode=MODE_INTERACTIVE_BINARY,
        source="bridge_full",
    )
    assert result["segvigen_contract_version"] == SEGVIGEN_CONTRACT_VERSION
    assert result["latent"] == "mock_output"  # backward-compat
    assert result["output_tex_slat"] == "mock_output"
    assert result["labels_source"] == "decoded_binary"
    assert result["mode"] == "interactive_binary"


def test_build_segvigen_seg_result_invalid_mode():
    from core.contracts import build_segvigen_seg_result
    with pytest.raises(ValueError, match="invalid mode"):
        build_segvigen_seg_result(mode="bogus")


def test_build_segvigen_seg_result_invalid_labels_source():
    from core.contracts import build_segvigen_seg_result
    with pytest.raises(ValueError, match="invalid labels_source"):
        build_segvigen_seg_result(labels_source="bogus")


def test_build_segvigen_cond():
    from core.contracts import build_segvigen_cond, SEGVIGEN_CONTRACT_VERSION
    cond = build_segvigen_cond("c512", "neg", cond_1024="c1024")
    assert cond["segvigen_contract_version"] == SEGVIGEN_CONTRACT_VERSION
    assert cond["cond_512"] == "c512"
    assert cond["neg_cond"] == "neg"
    assert cond["cond_1024"] == "c1024"


def test_build_segvigen_cond_no_1024():
    from core.contracts import build_segvigen_cond
    cond = build_segvigen_cond("c512", "neg")
    assert "cond_1024" not in cond


def test_build_segvigen_voxel():
    from core.contracts import build_segvigen_voxel, SEGVIGEN_CONTRACT_VERSION
    grid = np.zeros((32, 32, 32), dtype=bool)
    voxel = build_segvigen_voxel(grid, resolution=32)
    assert voxel["segvigen_contract_version"] == SEGVIGEN_CONTRACT_VERSION
    assert voxel["resolution"] == 32
    assert voxel["grid"] is grid
