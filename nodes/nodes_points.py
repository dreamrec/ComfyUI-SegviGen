"""
SegviGenPointInput: specify up to 10 voxel-space click coordinates.

Output SEGVIGEN_POINTS is a list of [x, y, z] int triples.
The max widget value is 127 (covers 128^3 grids).
Values are clamped to the actual grid size at runtime in encode_points_for_sampler.
"""
import logging

log = logging.getLogger("segvigen")


class SegviGenPointInput:
    """Define voxel-space click coordinates for interactive segmentation."""

    CATEGORY = "SegviGen"
    FUNCTION = "build_points"
    RETURN_TYPES = ("SEGVIGEN_POINTS",)
    RETURN_NAMES = ("points",)

    @classmethod
    def INPUT_TYPES(cls):
        coord_widget = lambda: ("INT", {
            "default": 32, "min": 0, "max": 127, "step": 1,
            "tooltip": "Voxel coordinate. Valid range: 0 to (voxel_resolution - 1).",
        })
        return {
            "required": {
                "num_points": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1,
                                       "tooltip": "Number of click points (1–10)"}),
            },
            "optional": {
                **{f"point_{i}_{ax}": coord_widget()
                   for i in range(1, 11) for ax in ("x", "y", "z")},
            },
        }

    def build_points(self, num_points: int = 1, **kwargs) -> tuple:
        points = []
        for i in range(1, num_points + 1):
            x = kwargs.get(f"point_{i}_x", 32)
            y = kwargs.get(f"point_{i}_y", 32)
            z = kwargs.get(f"point_{i}_z", 32)
            points.append([int(x), int(y), int(z)])
        log.debug(f"SegviGen: {num_points} click points: {points}")
        return (points,)
