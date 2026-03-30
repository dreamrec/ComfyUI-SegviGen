# Interactive Point-Based Segmentation — Design Spec (v2, post-audit)

## Problem

The current interactive sampler uses BFS flood-fill from click points, which only works for geometrically disconnected meshes. It cannot segment a single connected mesh (e.g., a tank into turret/hull/tracks/barrel). The original SegviGen paper achieves this via neural network inference with point-token conditioning.

## Verified Facts (from checkpoint inspection + code audit)

### Checkpoint
- **File**: `interactive_seg.ckpt` (7.86 GB, PyTorch Lightning format)
- **Format**: `ckpt['state_dict']` contains 641 keys prefixed with `gen3dseg.`
- **Flow model keys**: Identical to `full_seg.safetensors` (640 keys) — same architecture
- **Extra key**: `gen3dseg.seg_embeddings.weight` → `[1, 1536]` float32
- **Loading**: Strip `gen3dseg.` prefix, load flow_model via existing loader, load seg_embeddings separately
- **Channels**: `in_channels=64` (32 noise + 32 concat_cond), `out_channels=32`

### Token Interleaving (verified from original repo)
- Point tokens are **appended** to the SparseTensor (not inserted at specific positions)
- SparseTensor self-attention operates over all tokens with the same batch index — order doesn't matter
- After transformer blocks, strip point tokens by index (last 10 per batch element)
- Point tokens get the learned `seg_embeddings.weight` (active) or zeros (padding)
- Point coordinates: same space as voxel coords — no encoder mapping needed

### Texture Decoder (CRITICAL — audit finding)
- Texture decoder **requires** `subs` (subdivision guides from shape decoder)
- `subs` comes from `shape_slat_decoder(shape_slat, return_subs=True)`
- Our pipeline does NOT run the shape decoder — it creates SparseTensor directly from voxels
- **Resolution**: We must run TRELLIS2's shape decoder to get `subs`, OR skip the texture decoder entirely

### Chosen Approach: Skip Texture Decoder

Running the shape decoder adds ~10s and requires a shape SLAT (not just voxel occupancy). Instead:

1. The model outputs 32-channel latent features per voxel
2. For binary segmentation (white vs black), the latent should show bimodal distribution
3. **Use PCA → Otsu threshold** on the latent features to extract the binary mask
4. Falls back to K-means clustering if PCA doesn't produce clean separation
5. This avoids the subs dependency entirely

## Architecture

### Multi-Part Strategy (unchanged)

User clicks N points → sampler runs model N times (one per point) → N binary masks → merge.

```
Click 1 (turret)  → inference → PCA+threshold → mask 1
Click 2 (hull)    → inference → PCA+threshold → mask 2
Click 3 (tracks)  → inference → PCA+threshold → mask 3
                              ↓
                    Merge → labels [R,R,R]
                    Unassigned voxels → label N+1
```

### Batch Mode

`batch_mode` widget on sampler node:

- **`sequential`** (default): N separate inference runs. VRAM = 1× model. Time = N × ~15-20s.
- **`batched`**: Duplicate voxels N times in one batch. VRAM = 1× model + N× voxels. Time = ~20-25s total.

Note: with CFG, batched mode has effective batch size 2N. For 4 points with ~5000 voxels: 40,000 tokens in self-attention. Only for ≥16GB VRAM GPUs.

## Implementation

### Step 1: Checkpoint Loader

**File**: `nodes/nodes_sampler.py` — new `_load_interactive_checkpoint()`

```python
def _load_interactive_checkpoint(model_config, ckpt_path):
    """Load interactive checkpoint (PyTorch Lightning format)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]

    # Strip 'gen3dseg.' prefix
    flow_sd = {k.replace("gen3dseg.flow_model.", ""): v
               for k, v in sd.items() if "flow_model" in k}
    seg_embed = sd["gen3dseg.seg_embeddings.weight"]  # [1, 1536]

    # Load flow model (same architecture as full_seg)
    flow_model = _build_flow_model(model_config)
    flow_model.load_state_dict(flow_sd)

    return flow_model, seg_embed
```

### Step 2: Rewrite Gen3DSegInteractive

**File**: `core/interactive.py` — replace noise biasing with token interleaving

```python
class Gen3DSegInteractive(nn.Module):
    def __init__(self, flow_model, seg_embed_weight):
        super().__init__()
        self.flow_model = flow_model
        self.seg_embeddings = nn.Embedding(1, 1536)
        self.seg_embeddings.weight.data.copy_(seg_embed_weight)

    def forward(self, x, t, cond, input_points=None, **kwargs):
        concat_cond = kwargs.get("concat_cond")

        # 1. Build point embeddings SparseTensor
        point_embeds = self._build_point_tokens(input_points, x)

        # 2. Concat noise + shape condition
        if concat_cond is not None:
            h = sp.sparse_cat([x, concat_cond], dim=-1)
        else:
            h = x

        # 3. Input projection
        h = self.flow_model.input_layer(h)

        # 4. Append point tokens (same batch idx, actual 3D coords)
        n_voxels = len(h.coords)
        h = _append_point_tokens(h, point_embeds)

        # 5. Timestep + positional encoding + transformer blocks
        t_emb = self.flow_model.t_embedder(t)
        # ... adaLN, pos embedding, block loop ...
        for block in self.flow_model.blocks:
            h = block(h, t_emb, cond)

        # 6. Strip point tokens (keep only first n_voxels)
        h = _strip_point_tokens(h, n_voxels)

        # 7. Output projection
        h = self.flow_model.layer_norm(h)
        h = self.flow_model.out_layer(h)
        return h
```

### Step 3: Binary Mask Extraction

**File**: `core/interactive.py` — new `extract_binary_mask()`

```python
def extract_binary_mask(seg_latent_feats, coords, voxel_resolution):
    """Extract foreground mask from 32-channel latent via PCA + Otsu."""
    from sklearn.decomposition import PCA
    from skimage.filters import threshold_otsu

    # PCA to 1D — captures the dominant variation (white vs black)
    pca = PCA(n_components=1)
    scores = pca.fit_transform(feats).squeeze()  # [N]

    # Otsu threshold — finds optimal split between bimodal distribution
    thresh = threshold_otsu(scores)
    mask = scores > thresh

    return mask, scores  # mask=bool, scores=confidence
```

### Step 4: Sampler Loop

**File**: `nodes/nodes_sampler.py` — rewrite `SegviGenInteractiveSampler.sample()`

Sequential mode (per-point loop):
```python
masks = []
for i, point in enumerate(points):
    # Build single-point input
    input_points = _encode_single_point(point, voxel_resolution)

    # Run inference (reuse loaded model — no reload)
    result = sampler.sample(gen, noise.clone(), ..., input_points=input_points)

    # Extract binary mask from latent
    mask, confidence = extract_binary_mask(result.samples.feats, coords, vr)
    masks.append((mask, confidence))

# Merge N masks → label grid
labels = _merge_masks(masks, coords, vr)
```

### Step 5: Mask Merger

```python
def _merge_masks(masks, coords, R):
    """Merge N binary masks into multi-label grid [R,R,R]."""
    labels = np.zeros((R, R, R), dtype=np.int32)
    confidence = np.full((R, R, R), -np.inf)

    for label_idx, (mask, scores) in enumerate(masks, start=1):
        for j, (x, y, z) in enumerate(coords):
            if mask[j] and scores[j] > confidence[x, y, z]:
                labels[x, y, z] = label_idx
                confidence[x, y, z] = scores[j]

    # Unassigned occupied voxels → label N+1
    next_label = len(masks) + 1
    for j, (x, y, z) in enumerate(coords):
        if labels[x, y, z] == 0:
            labels[x, y, z] = next_label

    return labels
```

## Node Inputs (Interactive Sampler)

```
required:
  model_config      TRELLIS2_MODEL_CONFIG
  slat              SEGVIGEN_SLAT
  conditioning      SEGVIGEN_COND
  points            SEGVIGEN_POINTS

optional:
  trimesh           TRIMESH
  batch_mode        COMBO ["sequential", "batched"]  default="sequential"
  seed              INT       default=0
  steps             INT       default=12
  guidance_strength FLOAT     default=7.5
```

## Files Changed

| File | Change |
|------|--------|
| `install.py` | Download `interactive_seg.ckpt` from HF |
| `core/interactive.py` | Rewrite Gen3DSegInteractive: token interleaving + mask extraction |
| `nodes/nodes_sampler.py` | Rewrite InteractiveSampler: per-point loop + checkpoint loading |
| `nodes/comfy-env.toml` | Add `scikit-learn`, `scikit-image` |
| `requirements.txt` | Add `scikit-learn`, `scikit-image` |

## Risks (updated post-audit)

1. **PCA+Otsu may not work** if the model doesn't produce clean bimodal latents. Fallback: K-means with k=2.
2. **Token stripping by index** assumes point tokens are always at the end. Verify SparseTensor ordering is preserved through transformer blocks.
3. **adaLN_modulation and pos_embedder** access patterns in the monkey-patched forward need to exactly match the original SLatFlowModel.forward(). Subclassing may be safer than monkey-patching.
4. **torch.load security**: Use `weights_only=False` (required for Lightning checkpoints with non-tensor objects). The checkpoint is from a known HuggingFace repo.
5. **Resolution 64 vs 512**: The model's position embedder uses raw coords. Our voxels are in [0,63], the model was trained on [0,511]. May need `coords * 8` scaling. Test empirically.
