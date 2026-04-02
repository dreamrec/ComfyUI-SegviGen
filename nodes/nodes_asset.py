"""
Asset-native SegviGen encoding pipeline nodes.

These nodes implement the paper's original input path that starts from a
mesh/GLB asset rather than from the TRELLIS2 image-to-shape bridge:

    mesh/GLB -> texture preprocessing -> scene normalization ->
    flexible dual-grid voxelization -> .vxz -> shape/tex encoders -> SLAT

STATUS: Phase 3 stubs — correct interfaces, but raise NotImplementedError.
"""


class SegviGenAssetPrepare:
    """
    Prepare a mesh asset for SegviGen's native encoding pipeline.

    This is the paper's original input path:
    mesh/GLB -> texture preprocessing -> scene normalization ->
    flexible dual-grid voxelization -> .vxz serialization.

    STATUS: Stub — requires o_voxel library for dual-grid conversion.
    """

    CATEGORY = "SegviGen/Asset (coming soon)"
    FUNCTION = "prepare"
    RETURN_TYPES = ("SEGVIGEN_VOXEL",)
    RETURN_NAMES = ("voxel",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
            "optional": {
                "texture_resolution": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 256,
                    "tooltip": "Texture map resolution for preprocessing."}),
                "voxel_resolution": ("INT", {"default": 512, "min": 64, "max": 1024,
                    "tooltip": "Voxel grid resolution for dual-grid conversion."}),
                "normalize_scene": ("BOOLEAN", {"default": True,
                    "tooltip": "Normalize asset to [-0.5, 0.5] cube (paper default)."}),
            },
        }

    def prepare(self, trimesh, texture_resolution=1024, voxel_resolution=512, normalize_scene=True):
        raise NotImplementedError(
            "SegviGenAssetPrepare is not yet implemented. "
            "This node requires the o_voxel library for flexible dual-grid conversion. "
            "Use the TRELLIS2 bridge path (SegviGenVoxelEncode) instead."
        )


class SegviGenAssetEncode:
    """
    Encode a prepared asset into SEGVIGEN_SLAT using SegviGen's native encoders.

    This is the paper's original encoding: .vxz -> shape encoder -> tex encoder ->
    shape decoder -> SLAT with real shape_slat, tex_slat, and subs.

    STATUS: Stub — requires SegviGen's native encoder weights and o_voxel.
    """

    CATEGORY = "SegviGen/Asset (coming soon)"
    FUNCTION = "encode"
    RETURN_TYPES = ("SEGVIGEN_SLAT",)
    RETURN_NAMES = ("slat",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voxel": ("SEGVIGEN_VOXEL",),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def encode(self, voxel, seed=0):
        raise NotImplementedError(
            "SegviGenAssetEncode is not yet implemented. "
            "This node requires SegviGen's native shape/tex encoder weights. "
            "Use the TRELLIS2 bridge path (SegviGenVoxelEncode) instead."
        )
