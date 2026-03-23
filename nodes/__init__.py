from .nodes_voxel import SegviGenGLBtoVoxel, SegviGenVoxelEncode
from .nodes_preprocess import SegviGenPreprocess
from .nodes_conditioning import SegviGenGetConditioning
from .nodes_sampler import SegviGenFullSampler, SegviGenInteractiveSampler
from .nodes_points import SegviGenPointInput
from .nodes_output import SegviGenRenderPreview, SegviGenExportParts

NODE_CLASS_MAPPINGS = {
    "SegviGenGLBtoVoxel": SegviGenGLBtoVoxel,
    "SegviGenVoxelEncode": SegviGenVoxelEncode,
    "SegviGenPreprocess": SegviGenPreprocess,
    "SegviGenGetConditioning": SegviGenGetConditioning,
    "SegviGenFullSampler": SegviGenFullSampler,
    "SegviGenInteractiveSampler": SegviGenInteractiveSampler,
    "SegviGenPointInput": SegviGenPointInput,
    "SegviGenRenderPreview": SegviGenRenderPreview,
    "SegviGenExportParts": SegviGenExportParts,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SegviGenGLBtoVoxel": "SegviGen: GLB to Voxel",
    "SegviGenVoxelEncode": "SegviGen: Voxel Encode",
    "SegviGenPreprocess": "SegviGen: Preprocess (BiRefNet)",
    "SegviGenGetConditioning": "SegviGen: Get Conditioning",
    "SegviGenFullSampler": "SegviGen: Full Sampler",
    "SegviGenInteractiveSampler": "SegviGen: Interactive Sampler",
    "SegviGenPointInput": "SegviGen: Point Input",
    "SegviGenRenderPreview": "SegviGen: Render Preview",
    "SegviGenExportParts": "SegviGen: Export Parts",
}
