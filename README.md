# ComfyUI-SegviGen

ComfyUI nodes for [SegviGen](https://github.com/Nelipot-Lee/SegviGen) 3D part segmentation.

## Requirements

- ComfyUI-TRELLIS2 must be installed (provides trellis2 library + CUDA extensions)
- NVIDIA GPU with 24GB+ VRAM (RTX 3090 or better)

## Installation

1. Clone to `ComfyUI/custom_nodes/ComfyUI-SegviGen`
2. Restart ComfyUI — checkpoint downloads automatically on first use

## Nodes

| Node | Purpose |
|------|---------|
| SegviGen: Preprocess (BiRefNet) | Background removal |
| SegviGen: Get Conditioning | DinoV3 feature extraction |
| SegviGen: GLB to Voxel | Mesh → voxel grid |
| SegviGen: Voxel Encode | Voxel → SLAT latent |
| SegviGen: Full Sampler | Auto part segmentation |
| SegviGen: Interactive Sampler | Point-guided segmentation |
| SegviGen: Point Input | Define click coordinates |
| SegviGen: Render Preview | Colored segment preview |
| SegviGen: Export Parts | Per-part GLB export |

## Improvements over Aero-Ex port

- Interactive (point-guided) segmentation — ported from original
- `guidance_rescale` parameter restored
- Configurable texture size (256–4096) and face count
- Proper VRAM management (no singleton model manager)
- Progress bars + interrupt support
- Accurate mesh face count (100k default, not 50k)
