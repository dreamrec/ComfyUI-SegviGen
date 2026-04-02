"""
Guided segmentation conditioning nodes for SegviGen.

These nodes provide 2D-map-guided conditioning for SegviGen's full
segmentation pipeline — the user supplies a pre-segmented 2D color map
and the conditioning steers the 3D segmentation to match that layout.

STATUS: Phase 3 stubs — correct interfaces, but raise NotImplementedError.
"""


class SegviGenGet2DMapConditioning:
    """
    Extract 2D segmentation map conditioning for guided full segmentation.

    Accepts a pre-segmented 2D image (color map) where each color represents
    a desired part. The conditioning guides the 3D segmentation to match
    the 2D part layout.

    STATUS: Stub — requires full_seg_w_2d_map.ckpt checkpoint.
    """

    CATEGORY = "SegviGen/Guided (coming soon)"
    FUNCTION = "condition"
    RETURN_TYPES = ("SEGVIGEN_COND",)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_config": ("TRELLIS2_MODEL_CONFIG",),
                "segmentation_map": ("IMAGE", {
                    "tooltip": "2D segmentation map — each color = one desired 3D part."}),
            },
            "optional": {
                "mask": ("MASK",),
                "preserve_palette": ("BOOLEAN", {"default": True,
                    "tooltip": "Preserve input color palette in the 3D segmentation."}),
            },
        }

    def condition(self, model_config, segmentation_map, mask=None, preserve_palette=True):
        raise NotImplementedError(
            "SegviGenGet2DMapConditioning is not yet implemented. "
            "This node requires the full_seg_w_2d_map.ckpt checkpoint "
            "and 2D-guided conditioning extraction. "
            "Use SegviGenGetConditioning for standard conditioning."
        )
