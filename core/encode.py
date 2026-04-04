"""
SegviGen encode: extract shape data and sample tex_slat from TRELLIS2 internals.

Uses the stage shim pattern (from nodes_conditioning.py) to call TRELLIS2's
_sample_tex_slat() directly, producing real tex_slat without depending on
any TRELLIS2 node output type.
"""
import logging
import torch

log = logging.getLogger("segvigen")


def _get_stages():
    """Load TRELLIS2 stages module via the established shim pattern."""
    from core.trellis2_shim import load_trellis2_stages
    return load_trellis2_stages()


def _deserialize_sparse_tensor(data, device):
    """
    Deserialize a SparseTensor from an IPC dict or pass through a live one.

    Uses trellis2.modules.sparse.SparseTensor (not comfy.sparse) to avoid
    constructor incompatibilities (ComfyUI's SparseConvTensor rejects 'scale').
    """
    from trellis2.modules.sparse import SparseTensor

    if isinstance(data, dict) and data.get("_type") == "SparseTensor":
        feats = data["feats"].to(device=device, dtype=torch.float32)
        coords = data["coords"].to(device=device)
        return SparseTensor(feats=feats, coords=coords)
    elif hasattr(data, "feats") and hasattr(data, "coords"):
        return SparseTensor(
            feats=data.feats.to(device=device, dtype=torch.float32),
            coords=data.coords.to(device=device),
        )
    else:
        raise ValueError(f"SegviGen: unknown SparseTensor format: {type(data)}")


def _deserialize_subs(data, device):
    """Deserialize the subs list (list of SparseTensors or IPC dicts)."""
    if data is None:
        return None
    if isinstance(data, list):
        return [_deserialize_sparse_tensor(item, device) for item in data]
    # If it's a single IPC dict wrapping a list
    if isinstance(data, dict) and data.get("_type") == "list":
        return [_deserialize_sparse_tensor(item, device) for item in data.get("items", [])]
    # Try direct deserialization (might be an IPC-serialised structure)
    try:
        stages = _get_stages()
        stages._init_config()
        return stages._deserialize_from_ipc(data, device)
    except Exception as e:
        log.warning(f"SegviGen: subs deserialization failed ({e}); subs=None")
        return None


def extract_shape_data(shape_result: dict, device: str):
    """
    Extract shape_slat, subs, resolution, and pipeline_type from
    a TRELLIS2_SHAPE_RESULT dict.

    Returns:
        (shape_slat, subs, resolution, pipeline_type) tuple
    """
    torch_device = torch.device(device)
    shape_slat = _deserialize_sparse_tensor(shape_result["shape_slat"], torch_device)
    subs = _deserialize_subs(shape_result.get("subs"), torch_device)
    resolution = shape_result.get("resolution", 512)
    pipeline_type = shape_result.get("pipeline_type", "512")

    return shape_slat, subs, resolution, pipeline_type


def sample_tex_slat(
    shape_result: dict,
    conditioning: dict,
    device: str,
    dtype: "torch.dtype" = None,
    seed: int = 0,
    tex_guidance_strength: float = 7.5,
    tex_sampling_steps: int = 12,
):
    """
    Sample tex_slat using TRELLIS2's _sample_tex_slat() via stage shim.

    Args:
        shape_result: TRELLIS2_SHAPE_RESULT dict
        conditioning: TRELLIS2_CONDITIONING dict (cond_512, neg_cond, optionally cond_1024)
        device: torch device string
        dtype: compute dtype for noise (default: torch.float32, matching stages._DEFAULT_DTYPE)
        seed: random seed for texture sampling
        tex_guidance_strength: CFG strength for texture flow model
        tex_sampling_steps: number of sampling steps

    Returns:
        SparseTensor — denormalized tex_slat (as returned by _sample_tex_slat)
    """
    import comfy.model_management as mm

    stages = _get_stages()
    stages._init_config()

    torch_device = torch.device(device)
    compute_dtype = dtype if dtype is not None else torch.float32

    # Deserialize shape_slat using TRELLIS2's INTERNAL SparseTensor class
    # (from .trellis2.sparse, NOT trellis2.modules.sparse or comfy.sparse).
    #
    # Why: _sample_tex_slat does arithmetic like (shape_slat - mean) / std which
    # requires TRELLIS2's SparseTensor with operator support. We can't use:
    #   - stages._deserialize_from_ipc: passes scale= kwarg which ComfyUI's
    #     SparseConvTensor rejects (namespace collision in comfy.sparse)
    #   - our _deserialize_sparse_tensor: creates trellis2.modules.sparse.SparseTensor
    #     which lacks arithmetic operators
    #
    # Solution: import from the shim-loaded trellis2 subpackage directly.
    from trellis2.sparse import SparseTensor as _InternalST
    slat_data = shape_result["shape_slat"]
    if isinstance(slat_data, dict) and slat_data.get("_type") == "SparseTensor":
        _feats = slat_data["feats"].to(device=torch_device, dtype=torch.float32)
        _coords = slat_data["coords"].to(device=torch_device)
        shape_slat = _InternalST(feats=_feats, coords=_coords)
    elif hasattr(slat_data, "feats") and hasattr(slat_data, "coords"):
        shape_slat = _InternalST(
            feats=slat_data.feats.to(device=torch_device, dtype=torch.float32),
            coords=slat_data.coords.to(device=torch_device),
        )
    else:
        raise ValueError(f"SegviGen: cannot deserialize shape_slat for tex sampling: {type(slat_data)}")
    pipeline_type = shape_result.get("pipeline_type", "512")

    # Determine model key and conditioning based on pipeline type
    if pipeline_type == "512":
        model_key = "tex_slat_flow_model_512"
        cond_key = "cond_512"
    else:
        model_key = "tex_slat_flow_model_1024"
        if "cond_1024" in conditioning:
            cond_key = "cond_1024"
        else:
            log.warning(
                "SegviGen: pipeline_type=%s but cond_1024 not in conditioning; "
                "falling back to cond_512", pipeline_type
            )
            cond_key = "cond_512"

    # Move conditioning to device
    cond_tensor = conditioning[cond_key].to(device=torch_device, dtype=compute_dtype)
    neg_cond_tensor = conditioning["neg_cond"].to(device=torch_device, dtype=compute_dtype)
    tex_cond = {"cond": cond_tensor, "neg_cond": neg_cond_tensor}

    sampler_params = {
        "steps": tex_sampling_steps,
        "guidance_strength": tex_guidance_strength,
    }

    torch.manual_seed(seed)
    log.info(
        f"SegviGen: sampling tex_slat via stage shim "
        f"(model={model_key}, steps={tex_sampling_steps}, cfg={tex_guidance_strength})"
    )

    # _sample_tex_slat requires dtype as 6th positional arg (stages.py:524)
    tex_slat = stages._sample_tex_slat(
        tex_cond, model_key, shape_slat,
        sampler_params, torch_device, compute_dtype,
    )

    mm.soft_empty_cache()
    log.info(f"SegviGen: tex_slat sampled — {tex_slat.feats.shape[0]} tokens, "
             f"{tex_slat.feats.shape[1]} channels")
    return tex_slat
