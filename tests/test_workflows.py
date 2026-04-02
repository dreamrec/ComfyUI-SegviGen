"""Workflow JSON sanity tests — verify official workflows match code expectations."""
import json
import os
import pytest

WORKFLOWS_DIR = os.path.join(os.path.dirname(__file__), "..", "workflows")


def _load_workflow(name):
    path = os.path.join(WORKFLOWS_DIR, name)
    with open(path) as f:
        return json.load(f)


def _node_types_in_workflow(wf):
    return {n["type"] for n in wf["nodes"]}


def test_asset_native_uses_real_conditioning():
    """Asset-native workflow must use SegviGenGetConditioning, not NullConditioning."""
    wf = _load_workflow("segvigen_asset_native.json")
    types = _node_types_in_workflow(wf)
    assert "SegviGenGetConditioning" in types
    assert "SegviGenNullConditioning" not in types


def test_asset_native_has_asset_nodes():
    wf = _load_workflow("segvigen_asset_native.json")
    types = _node_types_in_workflow(wf)
    assert "SegviGenAssetPrepare" in types
    assert "SegviGenAssetEncode" in types


def test_guided_workflow_has_2d_map_conditioning():
    wf = _load_workflow("segvigen_2d_guided.json")
    types = _node_types_in_workflow(wf)
    assert "SegviGenGet2DMapConditioning" in types


def test_interactive_uses_voxel_encode():
    wf = _load_workflow("segvigen_interactive.json")
    types = _node_types_in_workflow(wf)
    assert "SegviGenVoxelEncode" in types
    assert "SegviGenFromShapeResult" not in types


def test_full_bridge_uses_voxel_encode():
    wf = _load_workflow("segvigen_image_conditioned.json")
    types = _node_types_in_workflow(wf)
    assert "SegviGenVoxelEncode" in types
    assert "SegviGenFromShapeResult" not in types


def test_no_official_workflow_uses_null_conditioning():
    """No official workflow should use SegviGenNullConditioning."""
    for name in os.listdir(WORKFLOWS_DIR):
        if not name.endswith(".json"):
            continue
        wf = _load_workflow(name)
        types = _node_types_in_workflow(wf)
        assert "SegviGenNullConditioning" not in types, (
            f"Workflow {name} uses SegviGenNullConditioning — "
            "official workflows should use real conditioning"
        )


def test_all_workflows_are_valid_json():
    for name in os.listdir(WORKFLOWS_DIR):
        if not name.endswith(".json"):
            continue
        path = os.path.join(WORKFLOWS_DIR, name)
        with open(path) as f:
            wf = json.load(f)
        assert "nodes" in wf, f"{name} missing 'nodes'"
        assert "links" in wf, f"{name} missing 'links'"
