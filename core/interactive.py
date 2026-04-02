"""
Gen3DSegInteractive: point-guided semantic segmentation via token interleaving.

Implements the original SegviGen paper's interactive segmentation mechanism:
- Point tokens are APPENDED to the SparseTensor self-attention sequence
- Each active point carries the learned seg_embeddings weight (1536-dim)
- After all transformer blocks, point tokens are stripped from the output
- The model produces binary colorization (selected part = white, rest = black)

This replaces the earlier noise-biasing approach which could not segment
connected meshes (e.g., a tank into turret/hull/tracks).
"""
import logging
import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger("segvigen")


# ─── Gen3DSegInteractive ────────────────────────────────────────────────────

class Gen3DSegInteractive(nn.Module):
    """
    SegviGen interactive segmentation model with point-token interleaving.

    Wraps a SLatFlowModel and injects point tokens into the self-attention
    sequence. Point tokens carry the learned seg_embeddings weight at the
    clicked voxel's 3D coordinates. After the transformer blocks, point
    tokens are stripped from the output.

    This matches the original paper's Gen3DSeg implementation:
    - Token interleaving (not noise biasing)
    - 10 fixed point slots (active + zero-padded)
    - Learned seg_embeddings.weight [1, 1536]
    """

    MAX_POINTS = 10

    def __init__(self, flow_model: nn.Module, seg_embed_weight: torch.Tensor):
        """
        Args:
            flow_model: SLatFlowModel instance (loaded from checkpoint)
            seg_embed_weight: [1, 1536] tensor — learned point embedding
        """
        super().__init__()
        self.flow_model = flow_model
        self.seg_embeddings = nn.Embedding(1, 1536)
        self.seg_embeddings.weight.data.copy_(seg_embed_weight)

    def _build_point_tokens(self, input_points, h):
        """
        Build a SparseTensor of 10 point tokens for injection.

        Args:
            input_points: dict with:
                - 'coords': [10, 4] int32 tensor (batch_idx, x, y, z)
                - 'labels': [10, 1] int32 tensor (1=active, 0=padding)
            h: the projected voxel SparseTensor (for device/dtype reference)

        Returns:
            SparseTensor with 10 tokens: active ones get seg_embeddings,
            padding ones get zeros. Features are 1536-dim (model_channels).
        """
        from trellis2.modules import sparse as sp

        device = h.feats.device
        dtype = h.feats.dtype
        model_ch = h.feats.shape[1]  # 1536 (after input_layer projection)

        coords = input_points['coords'].to(device)           # [10, 4]
        labels = input_points['labels'].squeeze(-1).to(device)  # [10]

        feats = torch.zeros(10, model_ch, device=device, dtype=dtype)
        active = labels == 1
        if active.any():
            embed = self.seg_embeddings.weight.to(dtype=dtype, device=device)  # [1, 1536]
            feats[active] = embed.expand(int(active.sum()), -1)

        return sp.SparseTensor(feats=feats, coords=coords)

    def forward(self, x, t, cond, input_points=None, **kwargs):
        """
        Forward pass with 2N token interleaving and point-token injection.

        Implements the upstream Gen3DSeg.forward() + flow_forward() structure:
          1. Build 2N tokens: [noise_0..N | tex_proxy_0..N] (block order)
          2. Concatenate with 2N concat_cond (duplicated shape_slat) → 2N × 64
          3. Project, add APE, inject 10 point tokens, run transformer
          4. Strip point tokens, extract first N rows (noise positions only)
          5. Output projection → [N, 32]

        Args:
            x: noisy sparse latent (SparseTensor, [N, 32])
            t: timestep tensor [B]
            cond: conditioning (tensor or list of tensors)
            input_points: dict with 'coords' [10,4] and 'labels' [10,1],
                          or None for unconditional forward
            **kwargs:
              concat_cond: normalized shape_slat SparseTensor [N, 32]
              tex_slats:   tex proxy SparseTensor [N, 32] — use zeros when
                           real tex_slat is unavailable (no tex encoder)
        """
        import torch.nn.functional as F
        from trellis2.modules import sparse as sp

        try:
            from trellis2.models.structured_latent_flow import manual_cast
        except ImportError:
            def manual_cast(x, dtype):
                if hasattr(x, "replace"):
                    return x.replace(x.feats.to(dtype))
                return x.to(dtype) if isinstance(x, torch.Tensor) else x

        concat_cond = kwargs.get("concat_cond")  # normalized shape_slat [N, 32]
        tex_slats   = kwargs.get("tex_slats")    # tex proxy [N, 32], or None
        fm = self.flow_model
        N = x.feats.shape[0]

        # ── 1. Build 2N token sequence ────────────────────────────────────
        # Upstream order: rows 0..N-1 = noise, rows N..2N-1 = tex_slat
        if tex_slats is not None:
            x_in = sp.SparseTensor(
                feats=torch.cat([x.feats, tex_slats.feats], dim=0),    # [2N, 32]
                coords=torch.cat([x.coords, tex_slats.coords], dim=0), # [2N, 4]
            )
            if concat_cond is not None:
                cc_in = sp.SparseTensor(
                    feats=torch.cat(
                        [concat_cond.feats, concat_cond.feats], dim=0
                    ),  # [2N, 32]
                    coords=torch.cat(
                        [concat_cond.coords, concat_cond.coords], dim=0
                    ),  # [2N, 4]
                )
            else:
                cc_in = None
        else:
            # Legacy path: N-token sequence (tex interleaving disabled)
            x_in = x
            cc_in = concat_cond

        n_tokens = x_in.feats.shape[0]  # 2N or N

        # ── 2. Feature concat: each token [32] + shape [32] → [64] ───────
        if cc_in is not None:
            x_in = sp.sparse_cat([x_in, cc_in], dim=-1)
        if isinstance(cond, list):
            cond = sp.VarLenTensor.from_tensor_list(cond)

        if x_in.feats.dtype != torch.float32:
            x_in = x_in.type(torch.float32)

        # ── 3. Input projection: [2N, 64] → [2N, 1536] ───────────────────
        h = fm.input_layer(x_in)
        h = manual_cast(h, fm.dtype)

        # ── 4. Timestep embedding ─────────────────────────────────────────
        t_emb = fm.t_embedder(t)
        if fm.share_mod:
            t_emb = fm.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, fm.dtype)
        cond  = manual_cast(cond, fm.dtype)

        # ── 5. Absolute position embedding (APE) ─────────────────────────
        if fm.pe_mode == "ape":
            pe = fm.pos_embedder(h.coords[:, 1:])
            h  = h + manual_cast(pe, fm.dtype)

        # ── 6. Append 10 point tokens after all 2N voxel tokens ──────────
        if input_points is not None:
            point_st = self._build_point_tokens(input_points, h)
            if fm.pe_mode == "ape":
                pt_pe    = fm.pos_embedder(point_st.coords[:, 1:])
                point_st = point_st.replace(
                    point_st.feats + manual_cast(pt_pe, fm.dtype)
                )
            combined_feats  = torch.cat([h.feats, point_st.feats],  dim=0)
            combined_coords = torch.cat([h.coords, point_st.coords], dim=0)
            h = sp.SparseTensor(feats=combined_feats, coords=combined_coords)

        # ── 7. Transformer blocks ─────────────────────────────────────────
        for block in fm.blocks:
            h = block(h, t_emb, cond)

        # ── 8. Strip point tokens (restore 2N voxel tokens) ──────────────
        if input_points is not None:
            h = sp.SparseTensor(
                feats=h.feats[:n_tokens],
                coords=h.coords[:n_tokens],
            )

        # ── 9. Extract noise positions: first N of 2N ─────────────────────
        # Rows N..2N-1 (tex positions) are discarded — we only need the
        # noise-position predictions for the Euler denoising step.
        if tex_slats is not None:
            h = sp.SparseTensor(
                feats=h.feats[:N],
                coords=h.coords[:N],
            )

        # ── 10. Output projection ─────────────────────────────────────────
        h = manual_cast(h, torch.float32)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = fm.out_layer(h)
        return h


# ─── Point encoding ─────────────────────────────────────────────────────────

def encode_single_point(
    point: list,
    voxel_resolution: int,
    device: str = "cpu",
) -> dict:
    """
    Encode a single click point as the input_points dict for Gen3DSegInteractive.

    Returns dict with:
        'coords': [10, 4] int32 — one active point + 9 zero-padded
        'labels': [10, 1] int32 — [1, 0, 0, ..., 0]
    """
    coords = torch.zeros(10, 4, dtype=torch.int32, device=device)
    labels = torch.zeros(10, 1, dtype=torch.int32, device=device)

    R = voxel_resolution
    px = max(0, min(R - 1, int(round(point[0]))))
    py = max(0, min(R - 1, int(round(point[1]))))
    pz = max(0, min(R - 1, int(round(point[2]))))

    coords[0] = torch.tensor([0, px, py, pz], dtype=torch.int32)
    labels[0] = 1

    return {'coords': coords.to(device), 'labels': labels.to(device)}


# ─── Multi-point packing ────────────────────────────────────────────────────

def pack_point_tokens(
    points_list: list,
    voxel_resolution: int,
    device: str = "cpu",
    slat_coords: "torch.Tensor" = None,
) -> dict:
    """
    Pack up to 10 click points into a single input_points dict for
    Gen3DSegInteractive.

    Upstream fidelity: the original inference_interactive.py maps click
    coordinates through tex_encoder to get latent-space coordinates. Since
    the bridge path has no tex_encoder, we approximate by snapping each
    click point to the nearest occupied coordinate in the SLAT's own
    coordinate space (slat_coords). This ensures point tokens are injected
    at valid latent positions rather than arbitrary voxel grid locations.

    Args:
        points_list: list of [x, y, z] click coordinates in 512-space
        voxel_resolution: coordinate space resolution (typically 512)
        device: torch device string
        slat_coords: [N, 4] int32 tensor of SLAT occupied coordinates
            (batch_idx, x, y, z). When provided, each click is snapped
            to the nearest occupied SLAT coordinate. When None, raw
            clamped coordinates are used (legacy behavior).

    Returns dict with:
        'coords': [10, 4] int32 — active points + zero-padded
        'labels': [10, 1] int32 — 1=active, 0=padding
    """
    MAX = Gen3DSegInteractive.MAX_POINTS  # 10
    if len(points_list) > MAX:
        log.warning(
            f"SegviGen: {len(points_list)} points provided, truncating to {MAX}"
        )
        points_list = points_list[:MAX]

    R = voxel_resolution
    coords = torch.zeros(MAX, 4, dtype=torch.int32, device=device)
    labels = torch.zeros(MAX, 1, dtype=torch.int32, device=device)

    # Pre-compute SLAT coordinate lookup for nearest-neighbor snapping
    slat_xyz = None
    if slat_coords is not None and slat_coords.shape[0] > 0:
        slat_xyz = slat_coords[:, 1:].float().cpu()  # [N, 3]

    for i, pt in enumerate(points_list):
        px = max(0, min(R - 1, int(round(pt[0]))))
        py = max(0, min(R - 1, int(round(pt[1]))))
        pz = max(0, min(R - 1, int(round(pt[2]))))

        if slat_xyz is not None:
            # Snap to nearest occupied SLAT coordinate (L2 distance)
            click = torch.tensor([px, py, pz], dtype=torch.float32)
            dists = (slat_xyz - click.unsqueeze(0)).pow(2).sum(dim=1)
            nearest_idx = int(dists.argmin())
            snapped = slat_coords[nearest_idx].to(dtype=torch.int32)
            coords[i] = snapped
            if (px, py, pz) != tuple(snapped[1:].tolist()):
                log.debug(
                    f"SegviGen: click ({px},{py},{pz}) snapped to "
                    f"SLAT coord ({snapped[1]},{snapped[2]},{snapped[3]})"
                )
        else:
            coords[i] = torch.tensor([0, px, py, pz], dtype=torch.int32)

        labels[i] = 1

    return {'coords': coords.to(device), 'labels': labels.to(device)}


# ─── K-means decode fallback ────────────────────────────────────────────────

def _decode_via_kmeans(seg_latent, coords_np, grid_resolution=64):
    """
    Decode segmentation latent to label grid via K-means clustering.

    Fallback path when _decode_tex_slat is unavailable.
    Extracted from SegviGenFullSampler.sample() lines 248-271.
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

    log.info(f"SegviGen: K-means decode — {k} clusters, "
             f"{len(np.unique(cluster_ids))} segments (grid {G}^3)")
    return labels


# ─── Tex-decode-based segmentation ─────────────────────────────────────────

def decode_seg_from_base_color(seg_latent, subs, coords_np,
                                voxel_resolution, grid_resolution=64):
    """
    Decode segmentation latent via _decode_tex_slat -> base_color thresholding.

    Primary path: denormalize seg_latent using tex_slat_normalization stats from
    pipeline.json, then call _decode_tex_slat(seg_latent, subs) to get PBR voxels,
    threshold base_color luminance for binary mask.
    Fallback: K-means on raw latent features.

    AUDIT C1: _decode_tex_slat expects DENORMALIZED input (stages.py applies
    slat * std + mean before decoding). The sampler output is normalized, so we
    must denormalize here before decoding.

    AUDIT H1: Runtime channel check — if seg_latent channels don't match the tex
    decoder's expected input, fall back to K-means immediately.
    """
    try:
        import torch
        from core.trellis2_shim import load_trellis2_stages
        stages = load_trellis2_stages()
        stages._init_config()

        # --- AUDIT H1: channel compatibility check ---
        expected_ch = stages._pipeline_config.get("models", {}).get(
            "tex_slat_flow_model_512", {}
        ).get("out_channels", None)
        actual_ch = seg_latent.feats.shape[1] if seg_latent.feats.dim() >= 2 else 0
        if expected_ch is not None and actual_ch != expected_ch:
            log.warning(
                "SegviGen: channel mismatch — seg_latent has %d channels, "
                "tex decoder expects %d; falling back to K-means",
                actual_ch, expected_ch,
            )
            return _decode_via_kmeans(seg_latent, coords_np, grid_resolution)

        # --- AUDIT C1: denormalize seg_latent before decoding ---
        denormed_latent = seg_latent  # default: pass through if no stats
        try:
            norm_stats = stages._pipeline_config.get("tex_slat_normalization")
            if norm_stats is not None:
                tex_mean = torch.tensor(
                    norm_stats["mean"], device=seg_latent.feats.device,
                    dtype=seg_latent.feats.dtype,
                )
                tex_std = torch.tensor(
                    norm_stats["std"], device=seg_latent.feats.device,
                    dtype=seg_latent.feats.dtype,
                )
                denormed_feats = seg_latent.feats * tex_std + tex_mean
                # Reconstruct SparseTensor with denormalized features
                from trellis2.representations import SparseTensor as _SparseTensor
                denormed_latent = _SparseTensor(
                    feats=denormed_feats, coords=seg_latent.coords,
                )
                log.debug("SegviGen: denormalized seg_latent for tex decode")
            else:
                log.warning(
                    "SegviGen: tex_slat_normalization not in pipeline.json; "
                    "passing raw latent to decoder (may produce incorrect colors)"
                )
        except Exception as e_norm:
            log.warning(
                "SegviGen: denormalization failed (%s); falling back to K-means",
                e_norm,
            )
            return _decode_via_kmeans(seg_latent, coords_np, grid_resolution)

        # _decode_tex_slat returns BATCHED SparseTensor with 6-ch PBR attrs
        # Must index [0] for batch 0 (same as run_texture_generation stages.py:963)
        # base_color = feats[:, 0:3], range [0, 1]
        decoded_batched = stages._decode_tex_slat(denormed_latent, subs)
        decoded = decoded_batched[0]  # batch 0
        base_color = decoded.feats[:, 0:3].cpu().float().numpy()

        # Luminance threshold: selected part = white, rest = black
        luminance = base_color.mean(axis=1)  # [N]
        mask = luminance > 0.5

        # Build label grid
        G = grid_resolution
        scale = voxel_resolution / G
        labels = np.zeros((G, G, G), dtype=np.int32)
        seg_coords = decoded.coords[:, 1:].cpu().numpy().astype(np.int32)
        scaled = np.clip((seg_coords / scale).astype(np.int32), 0, G - 1)

        for j in range(len(mask)):
            x, y, z = scaled[j]
            labels[x, y, z] = 1 if mask[j] else 2  # 1=selected, 2=rest

        log.info(f"SegviGen: decoded via _decode_tex_slat — "
                 f"{mask.sum()}/{len(mask)} voxels selected")
        return labels

    except Exception as e:
        log.warning(f"SegviGen: _decode_tex_slat failed ({e}); falling back to K-means")
        return _decode_via_kmeans(seg_latent, coords_np, grid_resolution)


# ─── Binary mask extraction ─────────────────────────────────────────────────

def extract_binary_mask(
    feats_np,
    coords_np=None,
    click_voxel=None,
    n_clusters: int = 8,
    fallback_k: int = 2,
):
    """
    Extract a binary foreground mask from 32-channel latent features.

    Two modes:

    **Spatial K-means mode** (preferred, when click_voxel + coords_np provided):
        Clusters the 32-dim features into n_clusters groups, then selects the
        cluster whose centroid voxel is nearest to the user's click point.
        This is robust even when the model output is not bimodal, because it
        uses the 3D click position to pick the semantically relevant cluster.

    **PCA+Otsu legacy mode** (fallback when no click position is available):
        PCA projects to 1D (dominant variation axis), Otsu thresholds for
        bimodal split. Falls back to K-means if the split is degenerate.

    Args:
        feats_np:    [N, C] float32 numpy array — per-voxel latent features
        coords_np:   [N, 3] int32 numpy array — voxel grid coordinates (optional)
        click_voxel: [3] array-like — click point in voxel grid space (optional)
        n_clusters:  number of K-means clusters for spatial mode
        fallback_k:  K-means cluster count for legacy PCA fallback

    Returns:
        (mask, scores): mask is bool [N], scores is float [N] (confidence)
    """
    # ── Spatial K-means mode (click-position-guided) ──────────────────────
    if click_voxel is not None and coords_np is not None:
        return _extract_mask_spatial(feats_np, coords_np, click_voxel, n_clusters)

    # ── PCA + Otsu legacy mode ────────────────────────────────────────────
    from sklearn.decomposition import PCA

    # PCA to 1D — captures the dominant variation (white vs black)
    pca = PCA(n_components=1)
    scores = pca.fit_transform(feats_np).squeeze()  # [N]

    # Bimodality check: gap ratio in PCA projection
    sorted_s = np.sort(scores)
    gap_ratio = np.diff(sorted_s).max() / max(sorted_s[-1] - sorted_s[0], 1e-8)

    if gap_ratio > 0.10:
        # Clearly bimodal — use Otsu threshold
        try:
            from skimage.filters import threshold_otsu
            thresh = threshold_otsu(scores)
        except (ImportError, ValueError):
            thresh = float(np.median(scores))
        mask = scores > thresh
        frac = mask.mean()
        # If >95% or <5%, threshold is bad — fall through to K-means
        if 0.05 <= frac <= 0.95:
            return mask, scores

    # PCA+Otsu failed (unimodal or degenerate) → K-means fallback
    log.warning(
        f"SegviGen: PCA+Otsu not suitable (gap_ratio={gap_ratio:.3f}), "
        f"falling back to K-means (k={fallback_k})"
    )
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=fallback_k, n_init=10, random_state=42)
    cluster_ids = km.fit_predict(feats_np)
    # Smaller cluster = foreground (clicked part is typically smaller)
    counts = [int((cluster_ids == c).sum()) for c in range(fallback_k)]
    fg_cluster = int(np.argmin(counts))
    mask = cluster_ids == fg_cluster
    # Confidence: negative distance to cluster center (closer = higher)
    scores = -np.linalg.norm(
        feats_np - km.cluster_centers_[fg_cluster], axis=1
    )
    return mask, scores


def _extract_mask_spatial(feats_np, coords_np, click_voxel, n_clusters=8):
    """
    Seed-based feature similarity mask extraction.

    Selects voxels whose 32-dim latent features are most similar to the
    voxel nearest to the click point.  This exploits the model's learned
    spatial feature coherence: voxels in the same semantic part tend to
    produce similar latent features, even when the global distribution is
    unimodal.

    Approach:
    1. Find the click-nearest voxel as the "seed"
    2. Compute L2 feature distance from all voxels to the seed
    3. Use click-guided K-means (k=2) with the seed's features as one
       centroid and the mean of spatially distant voxels as the other
    4. The cluster containing the seed = foreground

    This correctly handles both bimodal (cleanly segmented) and unimodal
    (spatially-coherent-but-not-separated) model outputs.

    Args:
        feats_np:    [N, C] float32 — per-voxel latent features
        coords_np:   [N, 3] int32 — voxel grid coordinates
        click_voxel: [3] — click position in voxel grid space
        n_clusters:  unused (kept for signature compatibility)

    Returns:
        (mask, scores): mask is bool [N], scores is float [N]
    """
    from sklearn.cluster import KMeans

    click = np.asarray(click_voxel, dtype=np.float32)
    coords_f = coords_np.astype(np.float32)

    # ── 1. Spatial distances to click (L1, fast) ─────────────────────
    spatial_dists = np.abs(coords_f - click).sum(axis=1)  # [N]

    # ── 2. Click-guided K-means init ─────────────────────────────────
    # Center 1: mean features of voxels near the click (near = bottom 20%)
    near_thr = np.percentile(spatial_dists, 20)
    near_mask = spatial_dists <= near_thr
    center_near = feats_np[near_mask].mean(axis=0)  # [C]

    # Center 2: mean features of voxels far from the click (top 20%)
    far_thr = np.percentile(spatial_dists, 80)
    far_mask = spatial_dists >= far_thr
    center_far = feats_np[far_mask].mean(axis=0)   # [C]

    # ── 3. K-means with 2 click-guided centroids ──────────────────────
    init_centers = np.vstack([center_near, center_far])
    km = KMeans(n_clusters=2, init=init_centers, n_init=1, random_state=42,
                max_iter=300)
    cluster_ids = km.fit_predict(feats_np)

    # ── 4. Foreground = cluster containing the click-nearest voxel ────
    seed_idx = int(np.argmin(spatial_dists))
    fg_cluster = int(cluster_ids[seed_idx])
    mask = cluster_ids == fg_cluster

    # Confidence: negative L2 distance to the foreground cluster center
    scores = -np.linalg.norm(feats_np - km.cluster_centers_[fg_cluster], axis=1)

    fg_pct = mask.mean() * 100
    log.info(
        f"SegviGen: click-guided K-means (k=2, seed_idx={seed_idx}, "
        f"spatial_dist={spatial_dists[seed_idx]:.1f}) → {fg_pct:.1f}% foreground"
    )
    return mask, scores


# ─── Multi-mask merger ───────────────────────────────────────────────────────

def merge_masks(masks_and_scores, coords_np, voxel_resolution, grid_resolution=64):
    """
    Merge N binary masks into a single multi-label [G,G,G] grid.

    Each voxel is assigned to the mask with highest confidence score.
    Unassigned occupied voxels get label N+1 ("the rest").

    Args:
        masks_and_scores: list of (mask_bool[N], scores_float[N]) tuples
        coords_np: [N, 3] int32 — voxel grid coordinates (in voxel_resolution space)
        voxel_resolution: int — coordinate space of coords_np (e.g. 512)
        grid_resolution: int — output label grid side length (default 64).
            coords are scaled from voxel_resolution → grid_resolution.
            Kept small (64) to avoid 512³ = 512 MB memory allocation.

    Returns:
        [G, G, G] int32 label grid (1-based labels, 0=empty)
    """
    G = grid_resolution
    # Scale factor: map coords from voxel_resolution space to grid_resolution space
    scale = voxel_resolution / G

    labels     = np.zeros((G, G, G), dtype=np.int32)
    confidence = np.full((G, G, G), -np.inf, dtype=np.float32)
    N = len(coords_np)

    # Vectorized scaling of coordinates
    scaled_coords = np.floor(coords_np.astype(np.float32) / scale).astype(np.int32)
    scaled_coords = np.clip(scaled_coords, 0, G - 1)

    # Assign each voxel to the mask with highest confidence
    for label_idx, (mask, scores) in enumerate(masks_and_scores, start=1):
        active = np.where(mask)[0]
        for j in active:
            x, y, z = scaled_coords[j, 0], scaled_coords[j, 1], scaled_coords[j, 2]
            if scores[j] > confidence[x, y, z]:
                labels[x, y, z] = label_idx
                confidence[x, y, z] = scores[j]

    # Unassigned occupied voxels → label N+1 ("the rest")
    next_label = len(masks_and_scores) + 1
    for j in range(N):
        x, y, z = scaled_coords[j, 0], scaled_coords[j, 1], scaled_coords[j, 2]
        if labels[x, y, z] == 0:
            labels[x, y, z] = next_label

    unique = np.unique(labels[labels > 0])
    log.info(
        f"SegviGen: merged {len(masks_and_scores)} masks → {len(unique)} labels "
        f"(grid {G}³, coord_scale={scale:.2f})"
    )
    return labels


# ─── Legacy helpers (backward compat) ────────────────────────────────────────

def bfs_labels_from_points(
    slat: dict,
    points_list: list,
    voxel_resolution: int,
) -> np.ndarray:
    """
    Build a [R,R,R] int32 label grid by BFS flood-fill from click points.

    LEGACY: Only works for geometrically disconnected meshes. Kept for
    backward compatibility. New code should use the model-based approach
    (Gen3DSegInteractive + extract_binary_mask + merge_masks).
    """
    from collections import deque

    raw_coords = slat["latent"].coords
    if isinstance(raw_coords, torch.Tensor):
        coords_np = raw_coords.detach().cpu().numpy().astype(np.int32)
    else:
        coords_np = np.asarray(raw_coords, dtype=np.int32)
    if coords_np.ndim == 2 and coords_np.shape[1] == 4:
        coords_np = coords_np[:, 1:]
    occupied = set(map(tuple, coords_np.tolist()))

    R = voxel_resolution
    labels_grid = np.zeros((R, R, R), dtype=np.int32)
    labeled: set = set()

    for label_idx, pt in enumerate(points_list[:10], start=1):
        px = max(0, min(R - 1, int(round(pt[0]))))
        py = max(0, min(R - 1, int(round(pt[1]))))
        pz = max(0, min(R - 1, int(round(pt[2]))))

        start = (px, py, pz)
        if start not in occupied:
            dists = np.abs(coords_np.astype(np.float32) - np.array([px, py, pz], dtype=np.float32)).sum(axis=1)
            nn = coords_np[int(dists.argmin())]
            start = (int(nn[0]), int(nn[1]), int(nn[2]))

        if start in labeled:
            continue

        visited: set = {start}
        queue: deque = deque([start])
        while queue:
            cx, cy, cz = queue.popleft()
            labeled.add((cx, cy, cz))
            labels_grid[cx, cy, cz] = label_idx
            for dx, dy, dz in (
                (1, 0, 0), (-1, 0, 0),
                (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1),
            ):
                nb = (cx + dx, cy + dy, cz + dz)
                if nb not in visited and nb in occupied:
                    visited.add(nb)
                    queue.append(nb)

    # Label remaining unclicked components
    remaining = occupied - labeled
    if remaining:
        next_label = label_idx + 1 if points_list else 1
        while remaining:
            seed = next(iter(remaining))
            comp_queue: deque = deque([seed])
            comp_visited: set = {seed}
            while comp_queue:
                cx, cy, cz = comp_queue.popleft()
                labels_grid[cx, cy, cz] = next_label
                labeled.add((cx, cy, cz))
                remaining.discard((cx, cy, cz))
                for dx, dy, dz in (
                    (1, 0, 0), (-1, 0, 0),
                    (0, 1, 0), (0, -1, 0),
                    (0, 0, 1), (0, 0, -1),
                ):
                    nb = (cx + dx, cy + dy, cz + dz)
                    if nb in remaining and nb not in comp_visited:
                        comp_visited.add(nb)
                        comp_queue.append(nb)
            next_label += 1

    return labels_grid


def encode_points_for_sampler(
    points_list: list,
    voxel_resolution: int = 64,
    max_points: int = 10,
    device: str = "cpu",
) -> torch.Tensor:
    """
    LEGACY: Convert SEGVIGEN_POINTS list to [B=1, N, 3] int tensor.
    New code should use encode_single_point() instead.
    """
    result = torch.zeros(1, max_points, 3, dtype=torch.int32)
    max_coord = voxel_resolution - 1
    for i, pt in enumerate(points_list[:max_points]):
        x, y, z = [max(0, min(int(c), max_coord)) for c in pt]
        result[0, i] = torch.tensor([x, y, z], dtype=torch.int32)
    return result.to(device)
