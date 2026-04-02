"""Tests for core/checkpoints.py — manifest and routing logic."""
import pytest


def test_manifest_has_required_modes():
    from core.checkpoints import CHECKPOINT_MANIFEST
    assert "interactive_binary" in CHECKPOINT_MANIFEST
    assert "full" in CHECKPOINT_MANIFEST
    assert "full_2d_guided" in CHECKPOINT_MANIFEST
    assert "full_legacy" in CHECKPOINT_MANIFEST


def test_manifest_entries_have_required_fields():
    from core.checkpoints import CHECKPOINT_MANIFEST
    required_fields = {"hf_repo", "filename", "format", "sha256", "description"}
    for mode, entry in CHECKPOINT_MANIFEST.items():
        for field in required_fields:
            assert field in entry, f"Mode '{mode}' missing field '{field}'"


def test_interactive_uses_fenghora():
    from core.checkpoints import CHECKPOINT_MANIFEST
    assert CHECKPOINT_MANIFEST["interactive_binary"]["hf_repo"] == "fenghora/SegviGen"


def test_full_uses_fenghora():
    from core.checkpoints import CHECKPOINT_MANIFEST
    assert CHECKPOINT_MANIFEST["full"]["hf_repo"] == "fenghora/SegviGen"


def test_legacy_uses_aero_ex():
    from core.checkpoints import CHECKPOINT_MANIFEST
    assert CHECKPOINT_MANIFEST["full_legacy"]["hf_repo"] == "Aero-Ex/SegviGen"


def test_verify_hash_none_always_passes():
    from core.checkpoints import verify_checkpoint_hash
    # When expected is None, always returns True
    assert verify_checkpoint_hash("/nonexistent/path", None) is True


def test_list_available_checkpoints_returns_dict():
    from core.checkpoints import list_available_checkpoints
    result = list_available_checkpoints()
    assert isinstance(result, dict)
    assert "interactive_binary" in result
    for mode, info in result.items():
        assert "exists" in info
        assert "manifest" in info
