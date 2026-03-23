"""
Core SegviGen inference pipeline.

Gen3DSeg: wraps the SegviGen flow model (a TRELLIS2 DiT fine-tuned for segmentation).
Sampler: FlowEuler guidance-interval sampler matching the original inference_full.py.

Based on: https://github.com/Nelipot-Lee/SegviGen/blob/main/inference_full.py
"""
import logging
import torch
import torch.nn as nn

log = logging.getLogger("segvigen")


class Gen3DSeg(nn.Module):
    """
    Wrapper around the SegviGen flow model checkpoint.

    The checkpoint is a TRELLIS2-format DiT fine-tuned for part segmentation.
    Loaded via TRELLIS2's model loading utilities.
    """

    def __init__(self, flow_model: nn.Module):
        super().__init__()
        self.flow_model = flow_model

    def forward(self, x, t, cond=None, **kwargs):
        """
        Single denoising forward pass.

        Args:
            x: noisy sparse latent [B, N, C]
            t: timestep tensor [B]
            cond: conditioning dict from SEGVIGEN_COND
            **kwargs: passed through to flow_model (e.g. transformer_options)
        """
        return self.flow_model(x, t, cond=cond, **kwargs)


class Sampler:
    """
    FlowEuler guidance-interval sampler for SegviGen.

    Implements classifier-free guidance with:
    - guidance_interval: [start, end] fraction of denoising steps where CFG is active
    - guidance_rescale: optional variance correction (0.0 = off)

    Based on original FlowEulerGuidanceIntervalSampler in inference_full.py.
    """

    def __init__(
        self,
        model: Gen3DSeg,
        steps: int = 12,
        guidance_strength: float = 7.5,
        guidance_rescale: float = 0.0,
        guidance_interval: tuple = (0.6, 0.9),
    ):
        self.model = model
        self.steps = steps
        self.guidance_strength = guidance_strength
        self.guidance_rescale = guidance_rescale
        self.guidance_interval = guidance_interval

    @torch.no_grad()
    def sample(
        self,
        x_init,
        cond,
        neg_cond,
        progress_callback=None,
        interrupt_check=None,
        model_kwargs: dict = None,
    ):
        """
        Run the full denoising loop.

        Args:
            x_init: initial noisy latent SparseTensor
            cond: positive conditioning (cond_512 or cond_1024 tensor)
            neg_cond: negative conditioning tensor
            progress_callback: callable(step) for progress reporting
            interrupt_check: callable() that raises if cancelled
            model_kwargs: extra keyword args forwarded to every model() call.
                Used by SegviGenInteractiveSampler to pass input_points to
                Gen3DSegInteractive.forward() without monkey-patching gen.forward.

        Returns:
            Denoised latent SparseTensor
        """
        mkw = model_kwargs or {}
        x = x_init
        timesteps = torch.linspace(1.0, 0.0, self.steps + 1)
        t_start, t_end = self.guidance_interval

        for i in range(self.steps):
            if interrupt_check:
                interrupt_check()

            t = timesteps[i]
            dt = timesteps[i] - timesteps[i + 1]
            t_frac = float(i) / self.steps

            use_cfg = t_start <= t_frac <= t_end

            t_batch = t.expand(x.shape[0] if hasattr(x, 'shape') else 1)

            if use_cfg:
                # Conditional + unconditional forward pass
                v_cond = self.model(x, t_batch, cond, **mkw)
                v_uncond = self.model(x, t_batch, neg_cond, **mkw)
                v = v_uncond + self.guidance_strength * (v_cond - v_uncond)

                if self.guidance_rescale > 0.0:
                    v = self._rescale(v, v_cond)
            else:
                v = self.model(x, t_batch, cond, **mkw)

            # Euler step
            x = x - dt * v

            if progress_callback:
                progress_callback(i + 1)

        return x

    def _rescale(self, v_guided, v_cond):
        """
        Rescale guided velocity to match conditional variance.
        Corrects over-sharpening from high guidance_strength.
        """
        std_cond = v_cond.std()
        std_guided = v_guided.std()
        if std_guided < 1e-8:
            return v_guided
        factor = std_cond / std_guided
        return v_guided * (self.guidance_rescale * factor + (1 - self.guidance_rescale))


def get_encoder_patcher(model_config: dict, models_dir: str):
    """
    Load the SegviGen shape SLAT encoder and return a ComfyUI ModelPatcher.

    The encoder model file is part of the SegviGen checkpoint bundle.
    """
    import os
    import safetensors.torch
    import comfy.model_management as mm
    import comfy.model_patcher

    # TODO: verify exact filename against Aero-Ex/SegviGen HuggingFace repo
    encoder_filename = "shape_vae_encoder.safetensors"
    encoder_path = os.path.join(models_dir, encoder_filename)

    if not os.path.exists(encoder_path):
        from install import ensure_encoder_checkpoint
        encoder_path = ensure_encoder_checkpoint(models_dir)

    from trellis2.models import SparseStructureVAE  # adjust import based on actual class
    dtype = model_config.get("dtype", torch.float16)
    encoder = SparseStructureVAE(encoder_only=True).to(dtype=dtype)
    state_dict = safetensors.torch.load_file(encoder_path)
    encoder.load_state_dict(state_dict, strict=False)
    encoder.eval()

    return comfy.model_patcher.ModelPatcher(encoder, load_device=mm.get_torch_device(),
                                             offload_device=mm.unet_offload_device())


def encode_voxel_to_slat(encoder, grid: "np.ndarray", model_config: dict) -> torch.Tensor:
    """
    Encode a boolean voxel grid to a SLAT latent tensor.

    Args:
        encoder: the loaded shape SLAT encoder model
        grid: np.ndarray bool [R, R, R] — the voxel occupancy grid
        model_config: TRELLIS2_MODEL_CONFIG dict

    Returns:
        Sparse SLAT latent (SparseTensor or equivalent)
    """
    import numpy as np
    import comfy.model_management as mm

    device = mm.get_torch_device()
    dtype = model_config.get("dtype", torch.float32)

    # Convert bool grid to voxel coordinates
    coords = np.argwhere(grid).astype(np.int32)  # [N, 3]
    if len(coords) == 0:
        raise ValueError("Voxel grid is empty — cannot encode")

    coords_tensor = torch.from_numpy(coords).unsqueeze(0).to(device)  # [1, N, 3]

    with torch.no_grad():
        latent = encoder.encode(coords_tensor)  # exact API TBD from encoder architecture

    return latent


def load_segvigen_checkpoint(model_config: dict, checkpoint_path: str) -> Gen3DSeg:
    """
    Load a SegviGen checkpoint into a Gen3DSeg wrapper.
    """
    import safetensors.torch
    from stages import get_flow_model  # ComfyUI-TRELLIS2/nodes/stages.py

    log.info(f"SegviGen: loading checkpoint from {checkpoint_path}")
    flow_model = get_flow_model(model_config)
    state_dict = safetensors.torch.load_file(checkpoint_path)
    flow_model.load_state_dict(state_dict, strict=False)
    return Gen3DSeg(flow_model)


def run_full_segmentation(
    model_config: dict,
    slat: dict,
    cond: dict,
    checkpoint_path: str,
    seed: int = 0,
    steps: int = 12,
    guidance_strength: float = 7.5,
    guidance_rescale: float = 0.0,
    guidance_interval_start: float = 0.6,
    guidance_interval_end: float = 0.9,
    progress_callback=None,
    interrupt_check=None,
) -> dict:
    """
    Run full (automatic) segmentation inference.

    Returns SEGVIGEN_SEG_RESULT dict.
    """
    import comfy.model_management as mm
    import comfy.model_patcher
    torch.manual_seed(seed)

    gen = load_segvigen_checkpoint(model_config, checkpoint_path)
    gen.eval()

    patcher = comfy.model_patcher.ModelPatcher(
        gen,
        load_device=mm.get_torch_device(),
        offload_device=mm.unet_offload_device(),
    )
    mm.load_models_gpu([patcher])

    sampler = Sampler(
        model=gen,
        steps=steps,
        guidance_strength=guidance_strength,
        guidance_rescale=guidance_rescale,
        guidance_interval=(guidance_interval_start, guidance_interval_end),
    )

    latent = slat["latent"]

    # Choose conditioning resolution: prefer cond_1024 if available
    pos_cond = cond.get("cond_1024", cond["cond_512"])
    neg_cond = cond["neg_cond"]

    # Add noise to latent (start from noisy version)
    x_init = _add_noise(latent, seed)

    result_latent = sampler.sample(
        x_init=x_init,
        cond=pos_cond,
        neg_cond=neg_cond,
        progress_callback=progress_callback,
        interrupt_check=interrupt_check,
    )

    # Decode latent → segment labels
    labels = _decode_to_labels(result_latent, slat)

    if labels.max() < 1:  # 0 = background; need at least label 1 for one part
        raise ValueError(
            "Segmentation returned 0 parts — try a different seed or increase guidance_strength"
        )

    return {
        "latent": result_latent,
        "labels": labels,
        "voxel": slat.get("voxel"),
        "mesh": None,  # filled by output nodes after GLB reconstruction
    }


def _add_noise(latent, seed: int):
    """Add flow-matching noise to start denoising from t=1."""
    torch.manual_seed(seed)
    if hasattr(latent, 'feats'):
        # SparseTensor — add noise to features
        import copy
        noisy = copy.copy(latent)
        noisy.feats = latent.feats + torch.randn_like(latent.feats)
        return noisy
    return latent + torch.randn_like(latent)


def _decode_to_labels(result_latent, slat: dict) -> "np.ndarray":
    """
    Decode the segmentation latent to per-voxel integer labels.

    Placeholder — actual decode uses trellis2 VAE decoder.
    """
    import numpy as np
    # Returns int32 array of shape [R, R, R]
    grid = slat.get("voxel", {}).get("grid")
    if grid is not None:
        return np.zeros(grid.shape, dtype=np.int32)
    return np.zeros((64, 64, 64), dtype=np.int32)
