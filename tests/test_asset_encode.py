"""Tests for core/asset_encode.py — o_voxel path discovery and resolution guards."""
import os
import sys
import types
import pytest


def test_o_voxel_env_override(tmp_path, monkeypatch):
    """SEGVIGEN_O_VOXEL_PATH env var should be the first candidate."""
    from core.asset_encode import _candidate_o_voxel_paths
    monkeypatch.setenv("SEGVIGEN_O_VOXEL_PATH", str(tmp_path))
    candidates = _candidate_o_voxel_paths("/fake/base")
    assert candidates[0] == str(tmp_path)


def test_o_voxel_windows_layout(tmp_path):
    """Windows Lib/site-packages layout should be in candidates."""
    from core.asset_encode import _candidate_o_voxel_paths
    candidates = _candidate_o_voxel_paths(str(tmp_path))
    win_path = os.path.join(
        str(tmp_path), "custom_nodes", "ComfyUI-TRELLIS2",
        "_env_trellis2", "Lib", "site-packages"
    )
    assert win_path in candidates


def test_o_voxel_linux_layout(tmp_path):
    """Linux lib/python*/site-packages should be checked via glob."""
    from core.asset_encode import _candidate_o_voxel_paths
    # Create a fake Linux layout
    linux_sp = tmp_path / "custom_nodes" / "ComfyUI-TRELLIS2" / "_env_trellis2" / "lib" / "python3.10" / "site-packages"
    linux_sp.mkdir(parents=True)
    candidates = _candidate_o_voxel_paths(str(tmp_path))
    assert str(linux_sp) in candidates


def test_o_voxel_missing_raises_actionable_error(monkeypatch):
    """When o_voxel is nowhere, error should list all checked paths."""
    from core.asset_encode import _ensure_o_voxel_path, _find_o_voxel_spec
    import core.asset_encode as mod

    # Reset cached path
    monkeypatch.setattr(mod, "_o_voxel_resolved_path", None)
    # Make find_spec return None
    monkeypatch.setattr(mod, "_find_o_voxel_spec", lambda: None)
    # Mock folder_paths
    fp = types.ModuleType("folder_paths")
    fp.base_path = "/nonexistent/comfyui"
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    # Ensure env var is not set
    monkeypatch.delenv("SEGVIGEN_O_VOXEL_PATH", raising=False)

    with pytest.raises(ImportError, match="o_voxel library not found"):
        _ensure_o_voxel_path()


def test_asset_resolution_512_required():
    """prepare_asset_to_vxz should reject non-512 resolution."""
    from core.asset_encode import prepare_asset_to_vxz
    with pytest.raises(ValueError, match="voxel_resolution=512"):
        prepare_asset_to_vxz("dummy", "/tmp/out.vxz", voxel_resolution=256)
