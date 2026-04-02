"""Tests for core/decode.py — label extraction from segmentation output."""
import numpy as np
import torch
import pytest


class _MockST:
    """Minimal SparseTensor stand-in."""
    def __init__(self, feats, coords):
        self.feats = feats
        self.coords = coords


def test_kmeans_fallback_produces_labels():
    from core.decode import kmeans_fallback
    N = 100
    seg_latent = _MockST(
        feats=torch.randn(N, 32),
        coords=torch.cat([
            torch.zeros(N, 1, dtype=torch.int32),
            torch.randint(0, 32, (N, 3), dtype=torch.int32),
        ], dim=1),
    )
    coords_np = seg_latent.coords[:, 1:].numpy().astype(np.int32)
    labels = kmeans_fallback(seg_latent, coords_np, grid_resolution=16)
    assert labels.shape == (16, 16, 16)
    assert labels.dtype == np.int32
    assert labels.max() > 0


def test_extract_binary_labels_from_mock_pbr():
    """Test binary label extraction from synthetic decoded PBR voxels."""
    from core.decode import extract_binary_labels

    N = 200
    # Simulate decoded voxels: first 100 are "white" (selected), rest are "black"
    feats = torch.zeros(N, 6)  # 6-ch PBR: base_color[0:3] + metallic + roughness + alpha
    feats[:100, 0:3] = 0.9  # white base_color
    feats[100:, 0:3] = 0.1  # black base_color

    coords = torch.cat([
        torch.zeros(N, 1, dtype=torch.int32),
        torch.randint(0, 32, (N, 3), dtype=torch.int32),
    ], dim=1)
    decoded = _MockST(feats=feats, coords=coords)

    labels = extract_binary_labels(decoded, voxel_resolution=32, grid_resolution=16)
    assert labels.shape == (16, 16, 16)
    # Should have at least label 1 (selected) and label 2 (remainder)
    unique = set(np.unique(labels)) - {0}
    assert len(unique) >= 1


def test_extract_color_cluster_labels_from_mock_pbr():
    """Test color-cluster label extraction from synthetic decoded PBR voxels."""
    from core.decode import extract_color_cluster_labels

    N = 300
    feats = torch.zeros(N, 6)
    # 3 distinct "colored" parts
    feats[:100, 0] = 0.9   # red
    feats[100:200, 1] = 0.9  # green
    feats[200:, 2] = 0.9    # blue

    coords = torch.cat([
        torch.zeros(N, 1, dtype=torch.int32),
        torch.randint(0, 32, (N, 3), dtype=torch.int32),
    ], dim=1)
    decoded = _MockST(feats=feats, coords=coords)

    labels = extract_color_cluster_labels(decoded, voxel_resolution=32, grid_resolution=16)
    assert labels.shape == (16, 16, 16)
    unique = set(np.unique(labels)) - {0}
    assert len(unique) >= 2  # at least 2 distinct parts


def test_decode_seg_result_interactive_fallback():
    """decode_seg_result should fall back to kmeans when subs is None."""
    from core.decode import decode_seg_result
    from core.contracts import LABELS_LATENT_KMEANS_FALLBACK

    N = 50
    seg_latent = _MockST(
        feats=torch.randn(N, 32),
        coords=torch.cat([
            torch.zeros(N, 1, dtype=torch.int32),
            torch.randint(0, 32, (N, 3), dtype=torch.int32),
        ], dim=1),
    )
    coords_np = seg_latent.coords[:, 1:].numpy().astype(np.int32)

    labels, labels_source, decoded = decode_seg_result(
        seg_latent, subs=None, coords_np=coords_np,
        voxel_resolution=32, mode="interactive_binary",
    )
    assert labels.shape[0] > 0
    assert labels_source == LABELS_LATENT_KMEANS_FALLBACK
    assert decoded is None


def test_decode_seg_result_full_fallback():
    """Full mode also falls back to kmeans when subs is None."""
    from core.decode import decode_seg_result
    from core.contracts import LABELS_LATENT_KMEANS_FALLBACK

    N = 50
    seg_latent = _MockST(
        feats=torch.randn(N, 32),
        coords=torch.cat([
            torch.zeros(N, 1, dtype=torch.int32),
            torch.randint(0, 32, (N, 3), dtype=torch.int32),
        ], dim=1),
    )
    coords_np = seg_latent.coords[:, 1:].numpy().astype(np.int32)

    labels, labels_source, decoded = decode_seg_result(
        seg_latent, subs=None, coords_np=coords_np,
        voxel_resolution=32, mode="full",
    )
    assert labels.max() > 0
    assert labels_source == LABELS_LATENT_KMEANS_FALLBACK
