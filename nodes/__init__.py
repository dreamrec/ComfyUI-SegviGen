from .nodes_loadmesh import SegviGenLoadMesh
from .nodes_voxel import SegviGenGLBtoVoxel, SegviGenVoxelEncode, SegviGenFromShapeResult
from .nodes_preprocess import SegviGenPreprocess
from .nodes_conditioning import SegviGenGetConditioning, SegviGenNullConditioning
from .nodes_sampler import SegviGenFullSampler, SegviGenInteractiveSampler
from .nodes_points import SegviGenPointInput
from .nodes_picker import SegviGenMeshPicker
from .nodes_output import (
    SegviGenRenderPreview,
    SegviGenExportParts,
)
from .nodes_asset import SegviGenAssetPrepare, SegviGenAssetEncode
from .nodes_guided import SegviGenGet2DMapConditioning

NODE_CLASS_MAPPINGS = {
    # ── Input / preprocessing ────────────────────────────────────────
    "SegviGenLoadMesh": SegviGenLoadMesh,
    "SegviGenPreprocess": SegviGenPreprocess,
    # ── Encoding ─────────────────────────────────────────────────────
    "SegviGenVoxelEncode": SegviGenVoxelEncode,
    "SegviGenAssetPrepare": SegviGenAssetPrepare,
    "SegviGenAssetEncode": SegviGenAssetEncode,
    # ── Conditioning ─────────────────────────────────────────────────
    "SegviGenGetConditioning": SegviGenGetConditioning,
    "SegviGenGet2DMapConditioning": SegviGenGet2DMapConditioning,
    # ── Sampling ─────────────────────────────────────────────────────
    "SegviGenFullSampler": SegviGenFullSampler,
    "SegviGenInteractiveSampler": SegviGenInteractiveSampler,
    # ── Points / picker ──────────────────────────────────────────────
    "SegviGenPointInput": SegviGenPointInput,
    "SegviGenMeshPicker": SegviGenMeshPicker,
    # ── Output ───────────────────────────────────────────────────────
    "SegviGenRenderPreview": SegviGenRenderPreview,
    "SegviGenExportParts": SegviGenExportParts,
    # ── Legacy (kept for backward compatibility) ─────────────────────
    "SegviGenGLBtoVoxel": SegviGenGLBtoVoxel,
    "SegviGenFromShapeResult": SegviGenFromShapeResult,
    "SegviGenNullConditioning": SegviGenNullConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # ── Input / preprocessing ────────────────────────────────────────
    "SegviGenLoadMesh": "SegviGen: Load Mesh",
    "SegviGenPreprocess": "SegviGen: Image Preprocessing",
    # ── Encoding ─────────────────────────────────────────────────────
    "SegviGenVoxelEncode": "SegviGen: Encode (shape + tex)",
    "SegviGenAssetPrepare": "SegviGen: Asset Prepare (coming soon)",
    "SegviGenAssetEncode": "SegviGen: Asset Encode (coming soon)",
    # ── Conditioning ─────────────────────────────────────────────────
    "SegviGenGetConditioning": "SegviGen: Conditioner",
    "SegviGenGet2DMapConditioning": "SegviGen: 2D Map Conditioning (coming soon)",
    # ── Sampling ─────────────────────────────────────────────────────
    "SegviGenFullSampler": "SegviGen: Full Sampling (experimental)",
    "SegviGenInteractiveSampler": "SegviGen: Interactive Sampler",
    # ── Points / picker ──────────────────────────────────────────────
    "SegviGenPointInput": "SegviGen: Point Input (512-space)",
    "SegviGenMeshPicker": "SegviGen: 3D Mesh Picker",
    # ── Output ───────────────────────────────────────────────────────
    "SegviGenRenderPreview": "SegviGen: Render Preview",
    "SegviGenExportParts": "SegviGen: Splitter",
    # ── Legacy ───────────────────────────────────────────────────────
    "SegviGenGLBtoVoxel": "SegviGen: Voxelizer (legacy occupancy)",
    "SegviGenFromShapeResult": "SegviGen: From TRELLIS2 Shape (legacy shape-only)",
    "SegviGenNullConditioning": "SegviGen: Null Conditioning (legacy, no image)",
}
