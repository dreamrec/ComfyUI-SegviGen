"""
SegviGen payload contracts — canonical constructors and validators for all
custom ComfyUI types exchanged between SegviGen nodes.

Every node MUST use these constructors to build output payloads. This
ensures consistent schema, explicit metadata, and forward-compatible
versioning across all paths (bridge, asset-native, legacy).

Contract version history:
  v3 (2026-04-02): initial freeze — source enum, labels_source, mode tags
"""
import logging

log = logging.getLogger("segvigen")

SEGVIGEN_CONTRACT_VERSION = 3

# ── Source enum values ────────────────────────────────────────────────────────
SOURCE_SHAPE_ONLY = "shape_only"
SOURCE_BRIDGE_FULL = "bridge_full"
SOURCE_ASSET_FULL = "asset_full"
_VALID_SOURCES = {SOURCE_SHAPE_ONLY, SOURCE_BRIDGE_FULL, SOURCE_ASSET_FULL}

# ── Labels source enum values ────────────────────────────────────────────────
LABELS_DECODED_BINARY = "decoded_binary"
LABELS_DECODED_COLOR_CLUSTER = "decoded_color_cluster"
LABELS_LATENT_KMEANS_FALLBACK = "latent_kmeans_fallback"
LABELS_NONE = "none"
_VALID_LABELS_SOURCES = {
    LABELS_DECODED_BINARY, LABELS_DECODED_COLOR_CLUSTER,
    LABELS_LATENT_KMEANS_FALLBACK, LABELS_NONE,
}

# ── Mode enum values ─────────────────────────────────────────────────────────
MODE_INTERACTIVE_BINARY = "interactive_binary"
MODE_FULL = "full"
MODE_FULL_2D_GUIDED = "full_2d_guided"
MODE_PREVIEW_PASSTHROUGH = "preview_passthrough"
_VALID_MODES = {
    MODE_INTERACTIVE_BINARY, MODE_FULL,
    MODE_FULL_2D_GUIDED, MODE_PREVIEW_PASSTHROUGH,
}


# ═══════════════════════════════════════════════════════════════════════════════
# SEGVIGEN_SLAT
# ═══════════════════════════════════════════════════════════════════════════════

def build_segvigen_slat(
    shape_slat,
    *,
    tex_slat=None,
    subs=None,
    voxel_resolution: int = 512,
    source: str = SOURCE_SHAPE_ONLY,
    pipeline_type: str = "512",
    normalization: dict = None,
):
    """
    Construct a SEGVIGEN_SLAT payload with guaranteed schema.

    Args:
        shape_slat: SparseTensor — required shape latent
        tex_slat: SparseTensor or None — texture latent (required for faithful sampling)
        subs: list[SparseTensor] or None — subdivision tensors (required for faithful decode)
        voxel_resolution: int — coordinate space resolution
        source: one of 'shape_only', 'bridge_full', 'asset_full'
        pipeline_type: '512', '1024', or '1536_cascade'
        normalization: optional dict with center, scale, resolution, coord_space

    Returns:
        dict — SEGVIGEN_SLAT payload
    """
    if source not in _VALID_SOURCES:
        raise ValueError(f"SegviGen contracts: invalid source '{source}', "
                         f"expected one of {_VALID_SOURCES}")

    return {
        "segvigen_contract_version": SEGVIGEN_CONTRACT_VERSION,
        "latent": shape_slat,       # backward-compatible alias
        "shape_slat": shape_slat,
        "tex_slat": tex_slat,
        "subs": subs,
        "voxel": {"resolution": voxel_resolution},
        "source": source,
        "pipeline_type": pipeline_type,
        "normalization": normalization,
    }


def validate_segvigen_slat(slat: dict, *, require_tex: bool = False):
    """
    Validate a SEGVIGEN_SLAT payload. Raises ValueError on invalid payloads.

    Args:
        slat: the payload dict to validate
        require_tex: if True, require tex_slat and subs to be non-None
    """
    if not isinstance(slat, dict):
        raise ValueError(f"SegviGen: SEGVIGEN_SLAT must be a dict, got {type(slat)}")

    shape = slat.get("shape_slat") or slat.get("latent")
    if shape is None:
        raise ValueError("SegviGen: SEGVIGEN_SLAT missing shape_slat/latent")

    source = slat.get("source")
    if source is not None and source not in _VALID_SOURCES:
        raise ValueError(f"SegviGen: invalid source '{source}' in SEGVIGEN_SLAT")

    if require_tex:
        if slat.get("tex_slat") is None:
            raise ValueError(
                f"SegviGen: faithful sampling requires tex_slat but SLAT has "
                f"source='{source}'. Connect SegviGenVoxelEncode with conditioning "
                f"to produce real tex_slat."
            )


def get_shape_slat(slat: dict):
    """Extract shape_slat from a SEGVIGEN_SLAT, handling both old and new keys."""
    return slat.get("shape_slat") or slat.get("latent")


# ═══════════════════════════════════════════════════════════════════════════════
# SEGVIGEN_SEG_RESULT
# ═══════════════════════════════════════════════════════════════════════════════

def build_segvigen_seg_result(
    *,
    output_tex_slat=None,
    decoded_tex_voxels=None,
    labels=None,
    labels_source: str = LABELS_NONE,
    mode: str = MODE_PREVIEW_PASSTHROUGH,
    mesh=None,
    voxel: dict = None,
    source: str = None,
):
    """
    Construct a SEGVIGEN_SEG_RESULT payload with guaranteed schema.

    Args:
        output_tex_slat: SparseTensor — sampler output (normalized)
        decoded_tex_voxels: decoded PBR voxels when available
        labels: np.ndarray int32 [G,G,G] label grid, or None
        labels_source: how labels were produced
        mode: sampling mode that produced this result
        mesh: original trimesh object
        voxel: metadata dict from SEGVIGEN_SLAT
        source: copied from SEGVIGEN_SLAT source field

    Returns:
        dict — SEGVIGEN_SEG_RESULT payload
    """
    if labels_source not in _VALID_LABELS_SOURCES:
        raise ValueError(f"SegviGen contracts: invalid labels_source '{labels_source}', "
                         f"expected one of {_VALID_LABELS_SOURCES}")
    if mode not in _VALID_MODES:
        raise ValueError(f"SegviGen contracts: invalid mode '{mode}', "
                         f"expected one of {_VALID_MODES}")

    return {
        "segvigen_contract_version": SEGVIGEN_CONTRACT_VERSION,
        "latent": output_tex_slat,          # backward-compatible alias
        "output_tex_slat": output_tex_slat,
        "decoded_tex_voxels": decoded_tex_voxels,
        "labels": labels,
        "labels_source": labels_source,
        "mode": mode,
        "mesh": mesh,
        "voxel": voxel,
        "source": source,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SEGVIGEN_COND
# ═══════════════════════════════════════════════════════════════════════════════

def build_segvigen_cond(
    cond_512,
    neg_cond,
    *,
    cond_1024=None,
):
    """
    Construct a SEGVIGEN_COND payload.

    Args:
        cond_512: torch.Tensor — DINOv3 512-pipeline conditioning
        neg_cond: torch.Tensor — negative/null conditioning
        cond_1024: torch.Tensor or None — 1024-pipeline conditioning

    Returns:
        dict — SEGVIGEN_COND payload
    """
    result = {
        "segvigen_contract_version": SEGVIGEN_CONTRACT_VERSION,
        "cond_512": cond_512,
        "neg_cond": neg_cond,
    }
    if cond_1024 is not None:
        result["cond_1024"] = cond_1024
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SEGVIGEN_VOXEL
# ═══════════════════════════════════════════════════════════════════════════════

def build_segvigen_voxel(
    grid,
    *,
    resolution: int,
    transform: dict = None,
):
    """
    Construct a SEGVIGEN_VOXEL payload.

    Args:
        grid: np.ndarray bool [R,R,R] — occupancy grid
        resolution: int — grid side length
        transform: optional normalization transform metadata

    Returns:
        dict — SEGVIGEN_VOXEL payload
    """
    return {
        "segvigen_contract_version": SEGVIGEN_CONTRACT_VERSION,
        "grid": grid,
        "resolution": resolution,
        "transform": transform,
    }
