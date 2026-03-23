from .nodes_voxel import SegviGenGLBtoVoxel, SegviGenVoxelEncode

NODE_CLASS_MAPPINGS = {
    "SegviGenGLBtoVoxel": SegviGenGLBtoVoxel,
    "SegviGenVoxelEncode": SegviGenVoxelEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SegviGenGLBtoVoxel": "SegviGen: GLB to Voxel",
    "SegviGenVoxelEncode": "SegviGen: Voxel Encode",
}
