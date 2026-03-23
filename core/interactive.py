"""
Gen3DSegInteractive: point-guided segmentation via forward_pre_hook injection.
"""
import logging
import torch
import torch.nn as nn
from core.pipeline import Gen3DSeg

log = logging.getLogger("segvigen")


class Gen3DSegInteractive(Gen3DSeg):
    """
    Gen3DSeg variant with learned point token injection.

    Point tokens are prepended to the sparse input sequence x before
    QKV projection in each SparseMultiHeadAttention block.
    """

    MAX_POINTS = 10

    def __init__(self, flow_model: nn.Module, voxel_resolution: int = 64):
        super().__init__(flow_model)
        # Learned embedding for the foreground point token
        # Dimension 1536 matches TRELLIS2's SparseMultiHeadAttention channel dim
        self.seg_embeddings = nn.Embedding(1, 1536)
        # Store resolution for positional encoding normalization
        self._voxel_resolution = voxel_resolution

    def get_positional_encoding(self, points: torch.Tensor) -> torch.Tensor:
        """
        Encode voxel-space [x,y,z] coords as transformer tokens.

        Args:
            points: [B, N, 3] int tensor — voxel coords (padded with 0 beyond num_points)
        Returns:
            [B, N, 1536] float tensor of point embeddings
        """
        B, N, _ = points.shape
        device = points.device

        # Normalize coords to [-1, 1] using actual voxel resolution
        voxel_resolution = getattr(self, "_voxel_resolution", 64)
        coords_norm = points.float() / (voxel_resolution - 1) * 2.0 - 1.0  # [B, N, 3]

        # Base embedding: learned foreground token repeated for each point
        base = self.seg_embeddings(
            torch.zeros(B, N, dtype=torch.long, device=device)
        )  # [B, N, 1536]

        # Sinusoidal positional encoding: 3 dims × (sin + cos) × 256 freqs = 6 × 256 = 1536
        freq = torch.arange(256, device=device).float()
        freq = torch.pow(10000.0, -freq / 256.0)  # [256]

        enc_parts = []
        for dim_idx in range(3):
            c = coords_norm[:, :, dim_idx:dim_idx+1]  # [B, N, 1]
            sins = torch.sin(c * freq.unsqueeze(0).unsqueeze(0))  # [B, N, 256]
            coss = torch.cos(c * freq.unsqueeze(0).unsqueeze(0))  # [B, N, 256]
            enc_parts.extend([sins, coss])

        # All 6 parts: 6 × 256 = 1536 channels — exact match to base embedding dim
        pos_enc = torch.cat(enc_parts, dim=-1)  # [B, N, 1536]

        return base + pos_enc

    def forward(self, x, t, cond=None, input_points: torch.Tensor = None, **kwargs):
        """
        Forward pass with point tokens injected into sparse attention.

        Args:
            x: noisy sparse latent
            t: timestep
            cond: conditioning dict
            input_points: [B, N, 3] int tensor of voxel-space click coordinates
        """
        if input_points is None:
            return super().forward(x, t, cond, **kwargs)

        point_embeds = self.get_positional_encoding(input_points)  # [B, N, 1536]

        # Collect all SparseMultiHeadAttention modules in the flow model
        attn_modules = _find_sparse_attn_modules(self.flow_model)
        log.debug(f"SegviGen interactive: injecting into {len(attn_modules)} attention blocks")

        hook_handles = []
        for attn_mod in attn_modules:
            handle = attn_mod.register_forward_pre_hook(
                lambda m, args, pe=point_embeds: _prepend_point_tokens(m, args, pe)
            )
            hook_handles.append(handle)

        try:
            result = super().forward(x, t, cond, **kwargs)
        finally:
            for h in hook_handles:
                h.remove()

        return result


def _find_sparse_attn_modules(model: nn.Module) -> list:
    """
    Walk the model tree and return all SparseMultiHeadAttention instances.
    """
    results = []
    for module in model.modules():
        if type(module).__name__ == "SparseMultiHeadAttention":
            results.append(module)
    return results


def _prepend_point_tokens(module, args, point_embeds: torch.Tensor):
    """
    forward_pre_hook: prepend point_embeds to the sparse input SparseTensor x.
    """
    x = args[0]

    try:
        # Strategy 1: SparseTensor concatenation
        from trellis2.sparse import SparseTensor, sparse_cat

        B, N_pts, C = point_embeds.shape

        dummy_coords = torch.zeros(B, N_pts, 3, dtype=torch.int32, device=point_embeds.device)
        point_sparse = SparseTensor(coords=dummy_coords, feats=point_embeds)
        x_augmented = sparse_cat([point_sparse, x], dim=1)
        log.debug(f"SegviGen: prepended {N_pts} point tokens via sparse_cat")
        return (x_augmented,) + args[1:]

    except Exception as e:
        # Strategy 2: fallback via transformer_options dict
        log.warning(f"SegviGen: sparse_cat failed ({e}), using transformer_options fallback")
        if len(args) > 2 and isinstance(args[-1], dict):
            args[-1]["segvigen_point_embeds"] = point_embeds
        return args


def encode_points_for_sampler(
    points_list: list,
    voxel_resolution: int = 64,
    max_points: int = 10,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Convert SEGVIGEN_POINTS list to the [B=1, N, 3] int tensor expected by
    Gen3DSegInteractive.forward().
    """
    result = torch.zeros(1, max_points, 3, dtype=torch.int32)
    max_coord = voxel_resolution - 1
    for i, pt in enumerate(points_list[:max_points]):
        x, y, z = [max(0, min(int(c), max_coord)) for c in pt]
        result[0, i] = torch.tensor([x, y, z], dtype=torch.int32)
    return result.to(device)
