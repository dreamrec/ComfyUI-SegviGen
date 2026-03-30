# Interactive Point-Based Segmentation — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable semantic part segmentation on connected meshes (e.g., tank → turret/hull/tracks/barrel) by properly implementing point-token interleaving in the flow model, using the dedicated `interactive_seg.ckpt` checkpoint.

**Architecture:** Each click point triggers a binary inference run (selected part = foreground). N clicks → N runs → N binary masks merged into a multi-label grid. Point tokens are appended to the SparseTensor self-attention sequence using the learned `seg_embeddings` weight from the interactive checkpoint. Binary masks extracted from the 32-channel latent output via PCA + Otsu thresholding.

**Tech Stack:** PyTorch, TRELLIS2 SparseTensor, scikit-learn (PCA), scikit-image (Otsu), ComfyUI node API

**Spec:** `docs/superpowers/specs/2026-03-26-interactive-segmentation-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `core/interactive.py` | `Gen3DSegInteractive` (token interleaving forward), `extract_binary_mask()`, `merge_masks()`. Keep `bfs_labels_from_points` and `encode_points_for_sampler` for backward compat. |
| `nodes/nodes_sampler.py` | `_load_interactive_checkpoint()`, rewritten `SegviGenInteractiveSampler.sample()` with per-point inference loop |
| `nodes/comfy-env.toml` | Add `scikit-learn`, `scikit-image` deps |
| `requirements.txt` | Add `scikit-learn`, `scikit-image` deps |
| `install.py` | Add `interactive_seg.ckpt` download |

---

## Chunk 1: Core Interactive Module

### Task 1: Rewrite Gen3DSegInteractive

**Files:**
- Modify: `core/interactive.py`

The key change: replace `_apply_point_bias` (noise biasing) with proper token interleaving that matches the original paper. The forward method must replicate `SLatFlowModel.forward()` exactly, with point tokens injected between `input_layer` and the block loop.

**Reference — SLatFlowModel.forward() at ComfyUI-TRELLIS2/nodes/trellis2/models/structured_latent_flow.py:169-199:**
```python
def forward(self, x, t, cond, concat_cond=None, **kwargs):
    if concat_cond is not None:
        x = sp.sparse_cat([x, concat_cond], dim=-1)  # feature concat
    if isinstance(cond, list):
        cond = sp.VarLenTensor.from_tensor_list(cond)
    h = self.input_layer(x)
    h = manual_cast(h, self.dtype)
    t_emb = self.t_embedder(t)
    if self.share_mod:
        t_emb = self.adaLN_modulation(t_emb)
    t_emb = manual_cast(t_emb, self.dtype)
    cond = manual_cast(cond, self.dtype)
    if self.pe_mode == "ape":
        pe = self.pos_embedder(h.coords[:, 1:])
        h = h + manual_cast(pe, self.dtype)
    for block in self.blocks:
        h = block(h, t_emb, cond)
    h = manual_cast(h, x.dtype)
    h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
    h = self.out_layer(h)
    return h
```

- [ ] **Step 1: Rewrite Gen3DSegInteractive class**

Replace the entire class in `core/interactive.py`. Keep `bfs_labels_from_points` and `encode_points_for_sampler` below it unchanged.

```python
class Gen3DSegInteractive(nn.Module):
    """
    SegviGen interactive segmentation model.

    Wraps a SLatFlowModel and injects point tokens into the self-attention
    sequence. Point tokens carry the learned seg_embeddings weight at the
    clicked voxel's coordinates. After the transformer blocks, point tokens
    are stripped from the output.

    This matches the original paper's Gen3DSeg implementation:
    - Token interleaving (not noise biasing)
    - 10 fixed point slots (active + zero-padded)
    - Learned seg_embeddings.weight [1, 1536]
    """

    MAX_POINTS = 10

    def __init__(self, flow_model: nn.Module, seg_embed_weight: torch.Tensor):
        super().__init__()
        self.flow_model = flow_model
        self.seg_embeddings = nn.Embedding(1, 1536)
        self.seg_embeddings.weight.data.copy_(seg_embed_weight)

    def _build_point_tokens(self, input_points, x):
        """
        Build a SparseTensor of 10 point tokens for injection.

        Args:
            input_points: dict with:
                - 'coords': [10, 4] int32 tensor (batch_idx, x, y, z)
                - 'labels': [10, 1] int32 tensor (1=active, 0=padding)
            x: the voxel SparseTensor (for device/dtype reference)

        Returns:
            SparseTensor with 10 tokens: active ones get seg_embeddings,
            padding ones get zeros. All have 1536-dim features (model_channels).
        """
        from trellis2.modules import sparse as sp

        device = x.feats.device
        dtype = x.feats.dtype
        model_ch = self.flow_model.input_layer.weight.shape[0]  # 1536

        coords = input_points['coords'].to(device)       # [10, 4]
        labels = input_points['labels'].squeeze(-1).to(device)  # [10]

        feats = torch.zeros(10, model_ch, device=device, dtype=dtype)
        active = labels == 1
        if active.any():
            embed = self.seg_embeddings.weight.to(dtype=dtype, device=device)  # [1, 1536]
            feats[active] = embed.expand(int(active.sum()), -1)

        return sp.SparseTensor(feats=feats, coords=coords)

    def forward(self, x, t, cond, input_points=None, **kwargs):
        """
        Forward pass with point-token interleaving.

        Replicates SLatFlowModel.forward() exactly, but appends point tokens
        after input_layer and strips them before out_layer.
        """
        import torch.nn.functional as F
        from trellis2.modules import sparse as sp
        from trellis2.models.structured_latent_flow import manual_cast

        concat_cond = kwargs.get("concat_cond")

        # ── 1. Feature concat (noise + shape condition) ──
        if concat_cond is not None:
            x = sp.sparse_cat([x, concat_cond], dim=-1)
        if isinstance(cond, list):
            cond = sp.VarLenTensor.from_tensor_list(cond)

        # ── 2. Input projection ──
        h = self.flow_model.input_layer(x)
        h = manual_cast(h, self.flow_model.dtype)

        # ── 3. Timestep embedding ──
        t_emb = self.flow_model.t_embedder(t)
        if self.flow_model.share_mod:
            t_emb = self.flow_model.adaLN_modulation(t_emb)
        t_emb = manual_cast(t_emb, self.flow_model.dtype)
        cond = manual_cast(cond, self.flow_model.dtype)

        # ── 4. Position embedding ──
        if self.flow_model.pe_mode == "ape":
            pe = self.flow_model.pos_embedder(h.coords[:, 1:])
            h = h + manual_cast(pe, self.flow_model.dtype)

        # ── 5. Append point tokens ──
        n_voxels = len(h.feats)
        if input_points is not None:
            point_st = self._build_point_tokens(input_points, h)
            # Also apply position embedding to point tokens
            if self.flow_model.pe_mode == "ape":
                pt_pe = self.flow_model.pos_embedder(point_st.coords[:, 1:])
                point_st = point_st.replace(
                    point_st.feats + manual_cast(pt_pe, self.flow_model.dtype)
                )
            # Append: concat feats and coords (same batch idx)
            combined_feats = torch.cat([h.feats, point_st.feats], dim=0)
            combined_coords = torch.cat([h.coords, point_st.coords], dim=0)
            h = sp.SparseTensor(feats=combined_feats, coords=combined_coords)

        # ── 6. Transformer blocks ──
        for block in self.flow_model.blocks:
            h = block(h, t_emb, cond)

        # ── 7. Strip point tokens (keep first n_voxels) ──
        if input_points is not None:
            h = sp.SparseTensor(
                feats=h.feats[:n_voxels],
                coords=h.coords[:n_voxels],
            )

        # ── 8. Output projection ──
        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.flow_model.out_layer(h)
        return h
```

- [ ] **Step 2: Add encode_single_point helper**

Add below Gen3DSegInteractive, before `bfs_labels_from_points`:

```python
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

    # Clamp point to valid range
    R = voxel_resolution
    px = max(0, min(R - 1, int(round(point[0]))))
    py = max(0, min(R - 1, int(round(point[1]))))
    pz = max(0, min(R - 1, int(round(point[2]))))

    coords[0] = torch.tensor([0, px, py, pz], dtype=torch.int32)
    labels[0] = 1

    return {'coords': coords.to(device), 'labels': labels.to(device)}
```

- [ ] **Step 3: Add extract_binary_mask function**

```python
def extract_binary_mask(feats_np, fallback_k=2):
    """
    Extract a binary foreground mask from 32-channel latent features.

    Uses PCA → Otsu thresholding. Falls back to K-means if PCA doesn't
    produce clean bimodal separation.

    Args:
        feats_np: [N, 32] float32 numpy array — per-voxel latent features
        fallback_k: cluster count for K-means fallback

    Returns:
        (mask, scores): mask is bool [N], scores is float [N] (confidence)
    """
    import numpy as np
    from sklearn.decomposition import PCA

    # PCA to 1D — captures the dominant variation (white vs black)
    pca = PCA(n_components=1)
    scores = pca.fit_transform(feats_np).squeeze()  # [N]

    # Try Otsu threshold
    try:
        from skimage.filters import threshold_otsu
        thresh = threshold_otsu(scores)
    except (ImportError, ValueError):
        # Fallback: simple median split
        thresh = float(np.median(scores))

    mask = scores > thresh

    # Sanity check: if mask is >90% or <10%, the split failed → use K-means
    frac = mask.mean()
    if frac < 0.05 or frac > 0.95:
        log.warning(f"SegviGen: PCA+Otsu produced {frac:.0%} foreground, "
                    f"falling back to K-means (k={fallback_k})")
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=fallback_k, n_init=10, random_state=42)
        cluster_ids = km.fit_predict(feats_np)
        # Smaller cluster = foreground (the clicked part is typically smaller)
        counts = [int((cluster_ids == c).sum()) for c in range(fallback_k)]
        fg_cluster = int(np.argmin(counts))
        mask = cluster_ids == fg_cluster
        scores = -np.linalg.norm(
            feats_np - km.cluster_centers_[fg_cluster], axis=1
        )  # negative distance = higher confidence

    return mask, scores
```

- [ ] **Step 4: Add merge_masks function**

```python
def merge_masks(masks_and_scores, coords_np, voxel_resolution):
    """
    Merge N binary masks into a single multi-label [R,R,R] grid.

    Args:
        masks_and_scores: list of (mask_bool[N], scores_float[N]) tuples
        coords_np: [N, 3] int32 — voxel coordinates
        voxel_resolution: int

    Returns:
        [R, R, R] int32 label grid (1-based, 0=empty)
    """
    import numpy as np

    R = voxel_resolution
    labels = np.zeros((R, R, R), dtype=np.int32)
    confidence = np.full((R, R, R), -np.inf, dtype=np.float32)
    N = len(coords_np)

    # Assign each voxel to the mask with highest confidence
    for label_idx, (mask, scores) in enumerate(masks_and_scores, start=1):
        for j in range(N):
            if mask[j]:
                x, y, z = int(coords_np[j, 0]), int(coords_np[j, 1]), int(coords_np[j, 2])
                if 0 <= x < R and 0 <= y < R and 0 <= z < R:
                    if scores[j] > confidence[x, y, z]:
                        labels[x, y, z] = label_idx
                        confidence[x, y, z] = scores[j]

    # Unassigned occupied voxels → label N+1 ("the rest")
    next_label = len(masks_and_scores) + 1
    for j in range(N):
        x, y, z = int(coords_np[j, 0]), int(coords_np[j, 1]), int(coords_np[j, 2])
        if 0 <= x < R and 0 <= y < R and 0 <= z < R:
            if labels[x, y, z] == 0:
                labels[x, y, z] = next_label

    unique = np.unique(labels[labels > 0])
    log.info(f"SegviGen: merged {len(masks_and_scores)} masks → {len(unique)} labels")
    return labels
```

- [ ] **Step 5: Commit core/interactive.py changes**

```bash
git add core/interactive.py
git commit -m "feat: rewrite Gen3DSegInteractive with token interleaving

Replace noise biasing with paper-correct point-token injection:
- Tokens appended to SparseTensor self-attention sequence
- seg_embeddings.weight loaded from interactive checkpoint
- PCA+Otsu binary mask extraction from 32-ch latent
- N-mask merger for multi-part segmentation"
```

---

## Chunk 2: Sampler Node + Checkpoint Loading

### Task 2: Interactive Checkpoint Loader

**Files:**
- Modify: `nodes/nodes_sampler.py`

- [ ] **Step 1: Add _load_interactive_checkpoint function**

Add after the existing `_load_segvigen_flow_model()` (~line 51):

```python
def _load_interactive_checkpoint(model_config: dict, ckpt_path: str):
    """
    Load the interactive segmentation checkpoint (PyTorch Lightning format).

    The .ckpt contains:
      - gen3dseg.flow_model.* — same SLatFlowModel as full_seg.safetensors
      - gen3dseg.seg_embeddings.weight — [1, 1536] learned point embedding

    Returns:
        (flow_model, seg_embed_weight) — flow_model wrapped in ComfyCompatFlowModel
    """
    import torch
    from core.pipeline import get_flow_model, ComfyCompatFlowModel

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    # Strip 'gen3dseg.flow_model.' prefix → bare SLatFlowModel keys
    flow_sd = {}
    for k, v in sd.items():
        if k.startswith("gen3dseg.flow_model."):
            flow_sd[k[len("gen3dseg.flow_model."):]] = v

    # Load flow model (same architecture as full_seg)
    flow_model = get_flow_model(model_config)
    flow_model.load_state_dict(flow_sd, strict=False)
    flow_model.eval()

    # Extract seg_embeddings weight
    seg_embed = sd["gen3dseg.seg_embeddings.weight"]  # [1, 1536]

    del ckpt, sd, flow_sd
    log.info(f"SegviGen: loaded interactive checkpoint from {ckpt_path}")
    log.info(f"  seg_embeddings: {seg_embed.shape}")

    return ComfyCompatFlowModel(flow_model), seg_embed
```

- [ ] **Step 2: Add _get_interactive_checkpoint_path function**

```python
def _get_interactive_checkpoint_path() -> str:
    """Return path to the interactive SegviGen checkpoint."""
    path = os.path.join(_get_models_dir(), "interactive_seg.ckpt")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"SegviGen interactive checkpoint not found: {path}\n"
            "Download it from https://huggingface.co/fenghora/SegviGen"
        )
    return path
```

### Task 3: Rewrite SegviGenInteractiveSampler.sample()

**Files:**
- Modify: `nodes/nodes_sampler.py`

- [ ] **Step 1: Update INPUT_TYPES to add batch_mode**

In `SegviGenInteractiveSampler.INPUT_TYPES()`, add to optional:

```python
"batch_mode": (["sequential", "batched"], {
    "default": "sequential",
    "tooltip": "sequential = low VRAM (N runs), batched = fast (1 run, more VRAM)",
}),
```

- [ ] **Step 2: Rewrite the sample() method**

Replace the entire `sample()` method body. Key changes:
1. Load `interactive_seg.ckpt` instead of `full_seg.safetensors`
2. Create `Gen3DSegInteractive` with the `seg_embed_weight`
3. Per-point inference loop (sequential mode)
4. Binary mask extraction per run
5. Merge masks into label grid

```python
def sample(self, model_config, slat, conditioning, points, trimesh=None,
           batch_mode="sequential", seed=0, steps=12,
           guidance_strength=7.5, guidance_rescale=0.0,
           guidance_interval_start=0.6, guidance_interval_end=0.9):
    import torch
    import numpy as np
    import comfy.model_management as mm
    import comfy.model_patcher
    from trellis2.samplers import FlowEulerGuidanceIntervalSampler
    from trellis2.modules import sparse as sp
    from core.interactive import (
        Gen3DSegInteractive, encode_single_point,
        extract_binary_mask, merge_masks,
    )

    check_interrupt()

    if not points or len(points) == 0:
        raise ValueError("SegviGen Interactive: no points provided. "
                         "Open the 3D picker and click at least one point.")

    # ── Load interactive checkpoint ──
    ckpt_path = _get_interactive_checkpoint_path()
    device = mm.get_torch_device()
    dtype_str = model_config.get("dtype", "fp16")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[dtype_str]

    log.info(f"SegviGen: loading interactive model from {ckpt_path}")
    flow_model_wrapped, seg_embed = _load_interactive_checkpoint(
        model_config, ckpt_path
    )

    # Create Gen3DSegInteractive with token interleaving
    gen = Gen3DSegInteractive(flow_model_wrapped.model, seg_embed)
    gen = gen.to(device=device, dtype=dtype)
    gen.eval()

    # Use ModelPatcher for memory management
    patcher = comfy.model_patcher.ModelPatcher(
        gen, load_device=device,
        offload_device=mm.unet_offload_device(),
    )
    mm.load_models_gpu([patcher])

    torch.manual_seed(seed)

    # ── Prepare noise and conditioning ──
    slat_latent = slat["latent"]
    voxel_resolution = (slat.get("voxel") or {}).get("resolution", 64)
    coords = slat_latent.coords
    coords_np = coords[:, 1:].cpu().numpy().astype(np.int32)

    noise_ch = gen.flow_model.out_channels   # 32
    cond_ch = gen.flow_model.in_channels - noise_ch  # 32

    pos_cond = conditioning.get("cond_1024", conditioning["cond_512"])
    neg_cond = conditioning["neg_cond"]

    def _to_device(c):
        if isinstance(c, torch.Tensor):
            return c.to(device=device, dtype=dtype)
        if isinstance(c, list):
            return [t.to(device=device, dtype=dtype)
                    if isinstance(t, torch.Tensor) else t for t in c]
        return c
    pos_cond = _to_device(pos_cond)
    neg_cond = _to_device(neg_cond)

    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)

    # ── Per-point inference loop ──
    masks_and_scores = []
    total = len(points)

    for i, point in enumerate(points):
        log.info(f"SegviGen interactive: point {i+1}/{total} at {point}")
        check_interrupt()

        # Fresh noise for each run
        noise = sp.SparseTensor(
            feats=torch.randn(len(coords), noise_ch,
                              device=device, dtype=dtype),
            coords=coords,
        )
        concat_cond = sp.SparseTensor(
            feats=torch.zeros(len(coords), cond_ch,
                              device=device, dtype=dtype),
            coords=coords,
        ) if cond_ch > 0 else None

        # Encode single point
        input_pts = encode_single_point(point, voxel_resolution,
                                        device=str(device))

        extra = {"concat_cond": concat_cond} if concat_cond else {}

        result = sampler.sample(
            gen, noise,
            cond=pos_cond, neg_cond=neg_cond,
            steps=steps,
            guidance_strength=guidance_strength,
            guidance_interval=(guidance_interval_start, guidance_interval_end),
            verbose=True,
            tqdm_desc=f"SegviGen pt {i+1}/{total}",
            input_points=input_pts,
            **extra,
        )

        # Extract binary mask from latent output
        feats_np = result.samples.feats.cpu().float().numpy()
        mask, scores = extract_binary_mask(feats_np)
        masks_and_scores.append((mask, scores))

        fg_pct = mask.mean() * 100
        log.info(f"SegviGen: point {i+1} → {fg_pct:.1f}% foreground")

    # ── Merge all masks into label grid ──
    labels = merge_masks(masks_and_scores, coords_np, voxel_resolution)

    mm.soft_empty_cache()
    return ({"latent": result.samples, "labels": labels,
             "voxel": slat.get("voxel"), "mesh": trimesh},)
```

- [ ] **Step 3: Commit sampler changes**

```bash
git add nodes/nodes_sampler.py
git commit -m "feat: rewrite InteractiveSampler with per-point inference loop

- Loads interactive_seg.ckpt (separate from full_seg.safetensors)
- Creates Gen3DSegInteractive with token interleaving
- Runs N inference passes (one per click point)
- Extracts binary masks via PCA+Otsu
- Merges into multi-label grid"
```

---

## Chunk 3: Dependencies + Install

### Task 4: Update dependencies

**Files:**
- Modify: `nodes/comfy-env.toml`
- Modify: `requirements.txt`

- [ ] **Step 1: Add scikit-learn and scikit-image to comfy-env.toml**

Already done (scikit-learn added). Add scikit-image:

```toml
# Under [pypi-dependencies]:
scikit-learn = "*"
scikit-image = "*"
```

- [ ] **Step 2: Add to requirements.txt**

```
easydict
scikit-learn
scikit-image
```

### Task 5: Update install.py

**Files:**
- Modify: `install.py`

- [ ] **Step 1: Add interactive checkpoint download**

Add `interactive_seg.ckpt` to the download list in `ensure_checkpoint()`. The checkpoint is at `huggingface.co/fenghora/SegviGen/interactive_seg.ckpt`.

- [ ] **Step 2: Commit dependency + install changes**

```bash
git add nodes/comfy-env.toml requirements.txt install.py
git commit -m "feat: add interactive checkpoint download + scikit deps"
```

---

## Chunk 4: Integration Testing

### Task 6: Smoke Test

- [ ] **Step 1: Restart ComfyUI**

- [ ] **Step 2: Load the interactive workflow**

Ctrl+O → `workflows/segvigen_interactive.json`

- [ ] **Step 3: Load the tank model**

Select `3d/cannon-generated.glb` in SegviGenLoadMesh

- [ ] **Step 4: Queue once to voxelize**

Click Run. Wait for voxelization + encoding to complete.

- [ ] **Step 5: Open picker, click 3-4 parts**

Click: turret body, barrel, tracks, hull (different semantic parts on the connected mesh).

- [ ] **Step 6: Queue again to run segmentation**

Watch console for:
```
SegviGen: loading interactive model from .../interactive_seg.ckpt
SegviGen interactive: point 1/4 at [x, y, z]
SegviGen: point 1 → XX.X% foreground
SegviGen interactive: point 2/4 at [x, y, z]
...
SegviGen: merged 4 masks → 5 labels
```

- [ ] **Step 7: Verify outputs**

Check:
1. Preview3D shows colored segments (different parts in different colors)
2. PreviewImage shows 2D rendered views with color-coded segments
3. Splitter exports combined GLB with named sub-objects
4. The tank body is NOT missing (all voxels accounted for)

- [ ] **Step 8: If PCA+Otsu fails (all same color)**

Check console for "PCA+Otsu produced XX% foreground, falling back to K-means". If K-means also fails, the model may need coordinate scaling (Risk 5 in spec). Try adding `coords * 8` scaling in `encode_single_point` to map from [0,63] to [0,504] range.

---

## Risk Mitigation Checklist

- [ ] Verify SparseTensor token ordering is preserved through transformer blocks (token stripping relies on this)
- [ ] Test with 1 point, 2 points, and 6+ points
- [ ] Test with disconnected mesh (robot) to verify backward compat
- [ ] Monitor VRAM usage during inference — report peak usage
- [ ] If model output is uniform (no bimodal split), try coordinate scaling: `coords[0, 1:] *= 8`
