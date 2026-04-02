"""
SegviGen decode module — ALL segmentation output decoding lives here.

Responsibilities:
  - Denormalize tex_slat using pipeline.json stats
  - Decode to PBR voxels via TRELLIS2 _decode_tex_slat
  - Extract binary labels (interactive mode)
  - Extract color-cluster labels (full mode)
  - Unified entry point for all decode paths
  - K-means fallback when decode fails
"""
import json
import logging
import os

import numpy as np

from core.contracts import (
    LABELS_DECODED_BINARY,
    LABELS_DECODED_COLOR_CLUSTER,
    LABELS_LATENT_KMEANS_FALLBACK,
)

log = logging.getLogger("segvigen")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Denormalize tex_slat
# ═══════════════════════════════════════════════════════════════════════════════

def denormalize_tex_slat(seg_latent, pipeline_json_path=None):
    """
    Read tex_slat_normalization from pipeline.json and apply
    ``feats * std + mean`` to produce denormalized features.

    Args:
        seg_latent: SparseTensor with normalized features
        pipeline_json_path: explicit path to pipeline.json, or None to
            auto-resolve via folder_paths.models_dir

    Returns:
        SparseTensor with denormalized features.  Falls back to returning
        the original seg_latent unchanged if stats are unavailable.
    """
    import torch

    # --- Resolve pipeline.json path ---
    if pipeline_json_path is None:
        try:
            import folder_paths
            pipeline_json_path = os.path.join(
                folder_paths.models_dir, "trellis2", "pipeline.json"
            )
        except ImportError:
            log.warning(
                "SegviGen decode: folder_paths unavailable; "
                "cannot locate pipeline.json — returning raw latent"
            )
            return seg_latent

    # --- Load normalization stats ---
    norm_stats = None
    try:
        if os.path.isfile(pipeline_json_path):
            with open(pipeline_json_path, "r") as f:
                pcfg = json.load(f)
            norm_stats = pcfg.get("tex_slat_normalization")
        else:
            log.warning(
                "SegviGen decode: pipeline.json not found at %s",
                pipeline_json_path,
            )
    except Exception as exc:
        log.warning(
            "SegviGen decode: failed to read pipeline.json (%s)", exc
        )

    if norm_stats is None:
        log.warning(
            "SegviGen decode: tex_slat_normalization not available; "
            "returning raw latent (colors may be incorrect)"
        )
        return seg_latent

    # --- Apply denormalization: feats * std + mean ---
    try:
        tex_mean = torch.tensor(
            norm_stats["mean"],
            device=seg_latent.feats.device,
            dtype=seg_latent.feats.dtype,
        )
        tex_std = torch.tensor(
            norm_stats["std"],
            device=seg_latent.feats.device,
            dtype=seg_latent.feats.dtype,
        )
        denormed_feats = seg_latent.feats * tex_std + tex_mean

        # Reconstruct SparseTensor with denormalized features
        SparseTensor = _get_sparse_tensor_class()
        denormed = SparseTensor(feats=denormed_feats, coords=seg_latent.coords)
        log.debug("SegviGen decode: denormalized seg_latent for tex decode")
        return denormed

    except Exception as exc:
        log.warning(
            "SegviGen decode: denormalization math failed (%s); "
            "returning raw latent", exc
        )
        return seg_latent


def _get_sparse_tensor_class():
    """
    Import SparseTensor, preferring trellis2.representations, falling back
    to trellis2.modules.sparse.
    """
    try:
        from trellis2.representations import SparseTensor
        return SparseTensor
    except (ImportError, ModuleNotFoundError):
        from trellis2.modules.sparse import SparseTensor
        return SparseTensor


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Decode to PBR voxels
# ═══════════════════════════════════════════════════════════════════════════════

def decode_to_pbr_voxels(denormed_latent, subs):
    """
    Decode a denormalized tex_slat to PBR voxels via TRELLIS2's _decode_tex_slat.

    Args:
        denormed_latent: SparseTensor — denormalized texture latent
        subs: list[SparseTensor] — subdivision tensors from the SLAT payload

    Returns:
        Decoded SparseTensor (batch 0) with PBR attributes.
        feats[:, 0:3] = base_color (RGB, range [0, 1]).
    """
    from core.trellis2_shim import load_trellis2_stages

    stages = load_trellis2_stages()
    stages._init_config()

    # _decode_tex_slat returns a BATCHED result; index [0] for batch 0
    decoded_batched = stages._decode_tex_slat(denormed_latent, subs)
    decoded = decoded_batched[0]

    log.debug(
        "SegviGen decode: PBR voxels decoded — %d voxels, %d channels",
        decoded.feats.shape[0], decoded.feats.shape[1],
    )
    return decoded


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Extract binary labels (interactive mode)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_binary_labels(decoded_voxels, voxel_resolution, grid_resolution=64):
    """
    Extract a binary label grid from decoded PBR voxels (interactive mode).

    Approach:
      - Extract base_color (feats[:, 0:3])
      - Compute luminance as mean of RGB channels
      - Threshold at 0.5
      - If selected ratio is degenerate (<5% or >95%), run Otsu on luminance
      - Map to [G,G,G] int32 grid: 1=selected, 2=remainder

    Args:
        decoded_voxels: SparseTensor with PBR features (at least 3 channels)
        voxel_resolution: int — coordinate space resolution
        grid_resolution: int — output label grid side length

    Returns:
        np.ndarray int32 [G, G, G] — 1=selected, 2=remainder, 0=empty
    """
    base_color = decoded_voxels.feats[:, 0:3].cpu().float().numpy()  # [N, 3]
    luminance = base_color.mean(axis=1)  # [N]

    # Initial threshold at 0.5
    mask = luminance > 0.5
    selected_ratio = mask.mean() if len(mask) > 0 else 0.0

    # Degenerate check: if <5% or >95%, use Otsu
    if selected_ratio < 0.05 or selected_ratio > 0.95:
        log.info(
            "SegviGen decode: binary threshold degenerate (%.1f%% selected), "
            "running Otsu fallback", selected_ratio * 100,
        )
        try:
            from skimage.filters import threshold_otsu
            thresh = threshold_otsu(luminance)
            mask = luminance > thresh
            log.info(
                "SegviGen decode: Otsu threshold=%.3f, %.1f%% selected",
                thresh, mask.mean() * 100,
            )
        except (ImportError, ValueError) as exc:
            log.warning(
                "SegviGen decode: Otsu failed (%s); using median threshold", exc
            )
            thresh = float(np.median(luminance))
            mask = luminance > thresh

    # Build [G,G,G] label grid
    G = grid_resolution
    scale = voxel_resolution / G
    labels = np.zeros((G, G, G), dtype=np.int32)
    seg_coords = decoded_voxels.coords[:, 1:].cpu().numpy().astype(np.int32)
    scaled = np.clip((seg_coords / scale).astype(np.int32), 0, G - 1)

    for j in range(len(mask)):
        x, y, z = scaled[j]
        labels[x, y, z] = 1 if mask[j] else 2

    n_selected = int(mask.sum())
    log.info(
        "SegviGen decode: binary labels — %d/%d voxels selected (grid %d^3)",
        n_selected, len(mask), G,
    )
    return labels


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Extract color-cluster labels (full mode)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_color_cluster_labels(
    decoded_voxels, voxel_resolution, grid_resolution=64, min_cluster_ratio=0.02
):
    """
    Extract multi-label grid from decoded PBR voxels via color clustering
    (full segmentation mode).

    Approach:
      1. Extract base_color (feats[:, 0:3])
      2. Convert to Lab color space (skimage.color.rgb2lab)
      3. Quantize via MiniBatchKMeans
      4. Merge tiny clusters (<min_cluster_ratio of total) into nearest large
      5. Run connected-component splitting inside each color cluster
      6. Sort final labels by descending voxel count

    Args:
        decoded_voxels: SparseTensor with PBR features
        voxel_resolution: int — coordinate space resolution
        grid_resolution: int — output label grid side length
        min_cluster_ratio: float — clusters smaller than this fraction get merged

    Returns:
        np.ndarray int32 [G, G, G] — 1-based labels, 0=empty
    """
    from sklearn.cluster import MiniBatchKMeans

    base_color = decoded_voxels.feats[:, 0:3].cpu().float().numpy()  # [N, 3]
    N = len(base_color)

    # --- Convert to Lab color space ---
    lab_features = _rgb_to_lab_safe(base_color)

    # --- Initial K-means quantization ---
    # Use k=8 as a reasonable starting point for color quantization
    n_init_clusters = min(8, max(2, N // 50))
    km = MiniBatchKMeans(
        n_clusters=n_init_clusters, n_init=5, random_state=0, batch_size=1024
    )
    cluster_ids = km.fit_predict(lab_features)
    centers = km.cluster_centers_.copy()

    # --- Merge tiny clusters into nearest large cluster ---
    cluster_ids, n_final_color = _merge_tiny_clusters(
        cluster_ids, centers, min_cluster_ratio
    )

    # --- Connected-component splitting within each color cluster ---
    seg_coords = decoded_voxels.coords[:, 1:].cpu().numpy().astype(np.int32)
    final_labels_per_voxel = _connected_component_split(cluster_ids, seg_coords)

    # --- Sort labels by descending voxel count ---
    final_labels_per_voxel = _sort_labels_by_count(final_labels_per_voxel)

    # --- Build [G,G,G] label grid ---
    G = grid_resolution
    scale = voxel_resolution / G
    labels = np.zeros((G, G, G), dtype=np.int32)
    scaled = np.clip((seg_coords / scale).astype(np.int32), 0, G - 1)

    for j in range(N):
        x, y, z = scaled[j]
        labels[x, y, z] = final_labels_per_voxel[j]

    unique_labels = np.unique(labels[labels > 0])
    log.info(
        "SegviGen decode: color-cluster labels — %d segments from %d voxels "
        "(grid %d^3, %d initial color clusters)",
        len(unique_labels), N, G, n_final_color,
    )
    return labels


def _rgb_to_lab_safe(base_color):
    """
    Convert RGB [N,3] in [0,1] to Lab color space. Falls back to RGB if
    skimage is unavailable.
    """
    try:
        from skimage.color import rgb2lab

        # rgb2lab expects [..., 3] with values in [0, 1]
        # Clamp to valid range
        rgb_clamped = np.clip(base_color, 0.0, 1.0)
        # rgb2lab needs at least 2D image-like input: reshape to [N, 1, 3]
        rgb_img = rgb_clamped.reshape(-1, 1, 3)
        lab_img = rgb2lab(rgb_img)  # [N, 1, 3]
        lab_features = lab_img.reshape(-1, 3)
        log.debug("SegviGen decode: converted RGB to Lab color space")
        return lab_features.astype(np.float32)

    except ImportError:
        log.warning(
            "SegviGen decode: skimage not available; using RGB for clustering "
            "(install scikit-image for better color segmentation)"
        )
        return base_color.astype(np.float32)


def _merge_tiny_clusters(cluster_ids, centers, min_ratio):
    """
    Merge clusters smaller than min_ratio of total into the nearest
    large cluster (by centroid L2 distance).

    Returns:
        (remapped_ids, n_large_clusters)
    """
    unique_ids = np.unique(cluster_ids)
    N = len(cluster_ids)
    min_count = int(N * min_ratio)

    # Identify large and small clusters
    counts = {cid: int((cluster_ids == cid).sum()) for cid in unique_ids}
    large = [cid for cid, cnt in counts.items() if cnt >= min_count]
    small = [cid for cid, cnt in counts.items() if cnt < min_count]

    if not large:
        # All clusters are tiny — keep them all as-is
        return cluster_ids, len(unique_ids)

    # Map each small cluster to the nearest large cluster
    remap = {}
    for s_cid in small:
        best_dist = np.inf
        best_large = large[0]
        for l_cid in large:
            dist = np.linalg.norm(centers[s_cid] - centers[l_cid])
            if dist < best_dist:
                best_dist = dist
                best_large = l_cid
        remap[s_cid] = best_large

    # Apply remapping
    if remap:
        remapped = cluster_ids.copy()
        for s_cid, l_cid in remap.items():
            remapped[cluster_ids == s_cid] = l_cid
        cluster_ids = remapped
        log.debug(
            "SegviGen decode: merged %d tiny clusters into %d large clusters",
            len(small), len(large),
        )

    return cluster_ids, len(large)


def _connected_component_split(cluster_ids, coords):
    """
    Run connected-component labeling within each color cluster.

    Uses 6-connectivity flood fill on the voxel grid. Each connected
    component within a color cluster becomes a separate final label.

    Args:
        cluster_ids: [N] int — color cluster assignment per voxel
        coords: [N, 3] int32 — voxel coordinates

    Returns:
        [N] int — 1-based final label per voxel
    """
    from collections import deque

    # Build coordinate -> index lookup
    coord_to_idx = {}
    for i in range(len(coords)):
        key = (int(coords[i, 0]), int(coords[i, 1]), int(coords[i, 2]))
        coord_to_idx[key] = i

    final_labels = np.zeros(len(cluster_ids), dtype=np.int32)
    visited = np.zeros(len(cluster_ids), dtype=bool)
    next_label = 1

    unique_clusters = np.unique(cluster_ids)
    neighbors_6 = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]

    for cid in unique_clusters:
        member_mask = cluster_ids == cid
        member_indices = np.where(member_mask)[0]

        for start_idx in member_indices:
            if visited[start_idx]:
                continue

            # BFS flood fill within this color cluster
            queue = deque([start_idx])
            visited[start_idx] = True
            component = [start_idx]

            while queue:
                cur = queue.popleft()
                cx, cy, cz = int(coords[cur, 0]), int(coords[cur, 1]), int(coords[cur, 2])
                for dx, dy, dz in neighbors_6:
                    nb_key = (cx + dx, cy + dy, cz + dz)
                    nb_idx = coord_to_idx.get(nb_key)
                    if nb_idx is not None and not visited[nb_idx] and cluster_ids[nb_idx] == cid:
                        visited[nb_idx] = True
                        queue.append(nb_idx)
                        component.append(nb_idx)

            for idx in component:
                final_labels[idx] = next_label
            next_label += 1

    return final_labels


def _sort_labels_by_count(labels):
    """
    Re-number labels so that label 1 has the most voxels, label 2 the
    second most, etc.
    """
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) == 0:
        return labels

    counts = [(lbl, int((labels == lbl).sum())) for lbl in unique_labels]
    counts.sort(key=lambda x: -x[1])  # descending by count

    remap = {old_lbl: new_lbl for new_lbl, (old_lbl, _) in enumerate(counts, start=1)}
    result = np.zeros_like(labels)
    for old_lbl, new_lbl in remap.items():
        result[labels == old_lbl] = new_lbl

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Unified decode entry point
# ═══════════════════════════════════════════════════════════════════════════════

def decode_seg_result(
    seg_latent, subs, coords_np, voxel_resolution, mode, grid_resolution=64
):
    """
    Unified entry point for all segmentation decode paths.

    Args:
        seg_latent: SparseTensor — sampler output (normalized)
        subs: list[SparseTensor] — subdivision tensors
        coords_np: [N, 3] int32 numpy array — voxel coordinates
        voxel_resolution: int — coordinate space resolution
        mode: str — "interactive_binary", "full", or "full_2d_guided"
        grid_resolution: int — output label grid side length

    Returns:
        (labels, labels_source, decoded_voxels_or_None)
          labels: np.ndarray int32 [G,G,G]
          labels_source: str — one of the LABELS_* constants
          decoded_voxels_or_None: decoded SparseTensor or None on fallback
    """
    if mode == "interactive_binary":
        return _decode_binary_path(
            seg_latent, subs, coords_np, voxel_resolution, grid_resolution
        )
    elif mode in ("full", "full_2d_guided"):
        return _decode_color_cluster_path(
            seg_latent, subs, coords_np, voxel_resolution, grid_resolution
        )
    else:
        log.warning(
            "SegviGen decode: unknown mode '%s', falling back to K-means", mode
        )
        labels = kmeans_fallback(seg_latent, coords_np, grid_resolution)
        return labels, LABELS_LATENT_KMEANS_FALLBACK, None


def _decode_binary_path(seg_latent, subs, coords_np, voxel_resolution, grid_resolution):
    """Binary decode path with try/except fallback to K-means."""
    try:
        denormed = denormalize_tex_slat(seg_latent)
        decoded = decode_to_pbr_voxels(denormed, subs)
        labels = extract_binary_labels(decoded, voxel_resolution, grid_resolution)
        return labels, LABELS_DECODED_BINARY, decoded
    except Exception as exc:
        log.warning(
            "SegviGen decode: binary decode failed (%s); falling back to K-means",
            exc,
        )
        labels = kmeans_fallback(seg_latent, coords_np, grid_resolution)
        return labels, LABELS_LATENT_KMEANS_FALLBACK, None


def _decode_color_cluster_path(
    seg_latent, subs, coords_np, voxel_resolution, grid_resolution
):
    """Color-cluster decode path with try/except fallback to K-means."""
    try:
        denormed = denormalize_tex_slat(seg_latent)
        decoded = decode_to_pbr_voxels(denormed, subs)
        labels = extract_color_cluster_labels(
            decoded, voxel_resolution, grid_resolution
        )
        return labels, LABELS_DECODED_COLOR_CLUSTER, decoded
    except Exception as exc:
        log.warning(
            "SegviGen decode: color-cluster decode failed (%s); "
            "falling back to K-means", exc,
        )
        labels = kmeans_fallback(seg_latent, coords_np, grid_resolution)
        return labels, LABELS_LATENT_KMEANS_FALLBACK, None


# ═══════════════════════════════════════════════════════════════════════════════
# 6. K-means fallback
# ═══════════════════════════════════════════════════════════════════════════════

def kmeans_fallback(seg_latent, coords_np, grid_resolution=64):
    """
    K-means fallback for when the full decode pipeline fails.

    Clusters raw latent features into segments and maps to a [G,G,G] grid.
    Same logic as the existing _decode_via_kmeans in interactive.py.

    Args:
        seg_latent: SparseTensor with raw latent features
        coords_np: [N, 3] int32 numpy array — voxel coordinates (unused,
            coordinates are taken from seg_latent.coords)
        grid_resolution: int — output label grid side length

    Returns:
        np.ndarray int32 [G, G, G] — 1-based labels, 0=empty
    """
    from sklearn.cluster import MiniBatchKMeans

    feats = seg_latent.feats.cpu().float().numpy()
    seg_coords = seg_latent.coords[:, 1:].cpu().numpy().astype(np.int32)
    G = grid_resolution

    n_parts = 4
    k = min(n_parts, max(2, len(feats) // 10))
    km = MiniBatchKMeans(n_clusters=k, n_init=5, random_state=0)
    cluster_ids = km.fit_predict(feats)

    # Build [G,G,G] label grid (1-based: cluster 0 -> label 1, etc.)
    vr = max(seg_coords.max() + 1, 1) if len(seg_coords) > 0 else G
    scale = vr / G
    labels = np.zeros((G, G, G), dtype=np.int32)
    for j, (x, y, z) in enumerate(seg_coords):
        gx = min(int(x / scale), G - 1)
        gy = min(int(y / scale), G - 1)
        gz = min(int(z / scale), G - 1)
        if gx >= 0 and gy >= 0 and gz >= 0:
            labels[gx, gy, gz] = int(cluster_ids[j]) + 1

    log.info(
        "SegviGen decode: K-means fallback — %d clusters, "
        "%d segments (grid %d^3)",
        k, len(np.unique(cluster_ids)), G,
    )
    return labels
