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

NODE_CLASS_MAPPINGS = {
    "SegviGenLoadMesh": SegviGenLoadMesh,
    "SegviGenGLBtoVoxel": SegviGenGLBtoVoxel,
    "SegviGenVoxelEncode": SegviGenVoxelEncode,
    "SegviGenFromShapeResult": SegviGenFromShapeResult,
    "SegviGenPreprocess": SegviGenPreprocess,
    "SegviGenGetConditioning": SegviGenGetConditioning,
    "SegviGenNullConditioning": SegviGenNullConditioning,
    "SegviGenFullSampler": SegviGenFullSampler,
    "SegviGenInteractiveSampler": SegviGenInteractiveSampler,
    "SegviGenPointInput": SegviGenPointInput,
    "SegviGenMeshPicker": SegviGenMeshPicker,
    "SegviGenRenderPreview": SegviGenRenderPreview,
    "SegviGenExportParts": SegviGenExportParts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SegviGenLoadMesh": "SegviGen: Load Mesh",
    "SegviGenGLBtoVoxel": "SegviGen: Voxelizer",
    "SegviGenVoxelEncode": "SegviGen: Encode (shape + tex)",
    "SegviGenFromShapeResult": "SegviGen: From TRELLIS2 Shape (shape only)",
    "SegviGenPreprocess": "SegviGen: Image Preprocessing",
    "SegviGenGetConditioning": "SegviGen: Conditioner",
    "SegviGenNullConditioning": "SegviGen: Null Conditioning (no image)",
    "SegviGenFullSampler": "SegviGen: Sampling",
    "SegviGenInteractiveSampler": "SegviGen: Interactive Sampler",
    "SegviGenPointInput": "SegviGen: Point Input",
    "SegviGenMeshPicker": "SegviGen: 3D Mesh Picker",
    "SegviGenRenderPreview": "SegviGen: Render Preview",
    "SegviGenExportParts": "SegviGen: Splitter",
}
