import torch
import pytest


def test_encode_points_basic():
    from core.interactive import encode_points_for_sampler
    pts = [[10, 20, 30], [5, 5, 5]]
    result = encode_points_for_sampler(pts, voxel_resolution=64, max_points=10)
    assert result.shape == (1, 10, 3)
    assert result[0, 0].tolist() == [10, 20, 30]
    assert result[0, 1].tolist() == [5, 5, 5]
    # Remaining slots zero-padded
    assert result[0, 2].sum() == 0


def test_encode_points_clamping():
    from core.interactive import encode_points_for_sampler
    pts = [[-5, 100, 200]]
    result = encode_points_for_sampler(pts, voxel_resolution=64, max_points=10)
    x, y, z = result[0, 0].tolist()
    assert x == 0     # clamped from -5
    assert y == 63    # clamped from 100
    assert z == 63    # clamped from 200


def test_encode_points_truncates_at_max():
    from core.interactive import encode_points_for_sampler
    pts = [[i, i, i] for i in range(15)]
    result = encode_points_for_sampler(pts, voxel_resolution=64, max_points=10)
    assert result.shape == (1, 10, 3)  # truncated to max_points


def test_encode_points_empty():
    from core.interactive import encode_points_for_sampler
    result = encode_points_for_sampler([], voxel_resolution=64, max_points=10)
    assert result.sum() == 0  # all zeros


# ─── pack_point_tokens tests ─────────────────────────────────────────────────

def test_pack_point_tokens_single_point():
    from core.interactive import pack_point_tokens
    result = pack_point_tokens([[10, 20, 30]], voxel_resolution=64)
    assert result['coords'].shape == (10, 4)
    assert result['labels'].shape == (10, 1)
    # First point active
    assert result['labels'][0, 0].item() == 1
    assert result['coords'][0].tolist() == [0, 10, 20, 30]
    # Rest zero-padded
    assert result['labels'][1:].sum().item() == 0


def test_pack_point_tokens_ten_points():
    from core.interactive import pack_point_tokens
    pts = [[i, i + 1, i + 2] for i in range(10)]
    result = pack_point_tokens(pts, voxel_resolution=64)
    assert result['labels'].sum().item() == 10  # all 10 active


def test_pack_point_tokens_truncates_at_10():
    from core.interactive import pack_point_tokens
    pts = [[i, i, i] for i in range(15)]
    result = pack_point_tokens(pts, voxel_resolution=64)
    # Should still be [10, 4] — truncated
    assert result['coords'].shape == (10, 4)
    assert result['labels'].sum().item() == 10


def test_pack_point_tokens_clamps_coords():
    from core.interactive import pack_point_tokens
    result = pack_point_tokens([[-5, 100, 200]], voxel_resolution=64)
    x, y, z = result['coords'][0, 1:].tolist()
    assert x == 0    # clamped from -5
    assert y == 63   # clamped from 100
    assert z == 63   # clamped from 200
