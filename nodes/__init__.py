from .nodes_voxel import SegviGenGLBtoVoxel, SegviGenVoxelEncode
from .nodes_preprocess import SegviGenPreprocess
from .nodes_conditioning import SegviGenGetConditioning

NODE_CLASS_MAPPINGS = {
    "SegviGenGLBtoVoxel": SegviGenGLBtoVoxel,
    "SegviGenVoxelEncode": SegviGenVoxelEncode,
    "SegviGenPreprocess": SegviGenPreprocess,
    "SegviGenGetConditioning": SegviGenGetConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SegviGenGLBtoVoxel": "SegviGen: GLB to Voxel",
    "SegviGenVoxelEncode": "SegviGen: Voxel Encode",
    "SegviGenPreprocess": "SegviGen: Preprocess (BiRefNet)",
    "SegviGenGetConditioning": "SegviGen: Get Conditioning",
}
