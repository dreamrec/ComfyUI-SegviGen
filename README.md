# ComfyUI-SegviGen

ComfyUI nodes for [SegviGen](https://github.com/Nelipot-Lee/SegviGen) 3D mesh part segmentation.

## Current State (v3 contract freeze, April 2026)

This repo is in active recovery toward **upstream SegviGen paper fidelity**. Here is what works today and what does not:

| Path | Status | Notes |
|------|--------|-------|
| **Interactive segmentation (bridge)** | Production | Uses real `tex_slat` via TRELLIS2 bridge, packed multi-click, decoded binary labels |
| **Full auto-segmentation** | Bridge-faithful | Real `tex_slat`, 2N interleaving via Gen3DSegInteractive, decoded color-cluster labels (K-means fallback when decode unavailable) |
| **Asset-native encoding** | Paper-native | Full `process_glb_to_vxz` + `vxz_to_latent_slat` pipeline via o_voxel + TRELLIS2 encoders. Requires o_voxel CUDA extensions |
| **2D-guided full segmentation** | Implemented | Palette-safe conditioning from 2D label maps, auto-switches to `full_seg_w_2d_map.ckpt` |

**Architecture:** The repo supports two input paths — **TRELLIS2 bridge** (image → shape → bridge tex_slat) and **asset-native** (mesh → o_voxel → native encoders). Both converge into the same sampler and export pipeline. A third mode, **2D-guided full segmentation**, uses a different checkpoint selected automatically from the conditioning task_mode.

**Checkpoints:** All faithful modes use `fenghora/SegviGen` — interactive_seg.ckpt (interactive), full_seg.ckpt (full auto), full_seg_w_2d_map.ckpt (2D-guided). The legacy Aero-Ex/SegviGen safetensors checkpoint is supported as a fallback only. All checkpoint sha256 hashes are verified at download time.

## Requirements

- ComfyUI-TRELLIS2 must be installed (provides trellis2 library + CUDA extensions)
- NVIDIA GPU with 24GB+ VRAM (RTX 3090 or better)

## Installation

1. Clone to `ComfyUI/custom_nodes/ComfyUI-SegviGen`
2. Restart ComfyUI -- checkpoint downloads automatically on first use

## Nodes

| Node | Purpose | Status |
|------|---------|--------|
| SegviGen: Image Preprocessing | Background removal via BiRefNet | Stable |
| SegviGen: Conditioner | DinoV3 feature extraction for sampler | Stable |
| SegviGen: Encode (shape + tex) | TRELLIS2 bridge encoder with real tex_slat | Stable |
| SegviGen: Interactive Sampler | Point-guided binary segmentation | Stable |
| SegviGen: 3D Mesh Picker | Click-to-select 3D UI | Stable |
| SegviGen: Point Input | Manual coordinate entry | Stable |
| SegviGen: Render Preview | Colored segment preview images | Stable |
| SegviGen: Splitter | Per-part GLB export | Stable |
| SegviGen: Full Sampling (experimental) | Auto part segmentation | Experimental -- uses K-means heuristic |
| SegviGen: From TRELLIS2 Shape (shape only) | Shape-only encode (no tex_slat) | Legacy -- degraded quality |
| SegviGen: Null Conditioning (no image) | Empty conditioning for testing | Legacy -- not for production |
| SegviGen: Voxelizer | Mesh to occupancy voxel grid | Legacy |
| SegviGen: Load Mesh | Load mesh from file | Utility |

## Workflows

- `workflows/segvigen_interactive.json` -- Interactive point-guided segmentation (recommended)
- `workflows/segvigen_image_conditioned.json` -- Full auto-segmentation (experimental)
