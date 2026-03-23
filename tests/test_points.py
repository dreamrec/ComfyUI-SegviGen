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
