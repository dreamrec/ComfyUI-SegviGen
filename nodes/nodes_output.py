"""
SegviGen output nodes:
  - SegviGenRenderPreview: colored segment preview images
  - SegviGenExportParts: per-segment GLB files
"""
import logging
import os
from datetime import datetime

from .helpers import check_interrupt

log = logging.getLogger("segvigen")


class SegviGenRenderPreview:
    """Render colored preview images of the segmentation result."""

    CATEGORY = "SegviGen"
    FUNCTION = "render"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview_images",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seg_result": ("SEGVIGEN_SEG_RESULT",),
            },
            "optional": {
                "num_views": ("INT", {"default": 8, "min": 1, "max": 36, "step": 1}),
                "resolution": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 128}),
            },
        }

    def render(self, seg_result: dict, num_views: int = 8, resolution: int = 512):
        import comfy.model_management as mm
        from core.renderer import render_segmentation_preview

        check_interrupt()
        log.info(f"SegviGen: rendering {num_views} preview views at {resolution}px")

        images = render_segmentation_preview(
            seg_result=seg_result,
            num_views=num_views,
            resolution=resolution,
        )

        mm.soft_empty_cache()
        return (images,)


class SegviGenExportParts:
    """Export segmented mesh parts as individual GLB files."""

    CATEGORY = "SegviGen"
    FUNCTION = "export"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_paths",)

    OUTPUT_NODE = True  # tells ComfyUI this node has side effects

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seg_result": ("SEGVIGEN_SEG_RESULT",),
            },
            "optional": {
                "texture_size": (
                    ["256", "512", "1024", "2048", "4096"],
                    {"default": "2048",
                     "tooltip": "UV texture atlas resolution for exported GLBs"},
                ),
                "max_faces": ("INT", {
                    "default": 100_000, "min": 10_000, "max": 500_000, "step": 10_000,
                }),
                "min_segment_faces": ("INT", {
                    "default": 50, "min": 1, "max": 10_000, "step": 10,
                    "tooltip": "Discard segments smaller than this face count.",
                }),
            },
        }

    def export(
        self,
        seg_result: dict,
        texture_size: str = "2048",
        max_faces: int = 100_000,
        min_segment_faces: int = 50,
    ):
        import folder_paths
        from core.split import split_mesh_by_labels

        check_interrupt()

        mesh = seg_result.get("mesh")
        labels = seg_result.get("labels")

        if mesh is None:
            log.warning("SegviGen export: no mesh in seg_result, skipping export")
            return ("",)

        # Output directory: ComfyUI/output/segvigen/<timestamp>/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(folder_paths.output_directory, "segvigen", timestamp)
        os.makedirs(out_dir, exist_ok=True)

        # Convert per-voxel labels to per-face labels
        face_labels = _voxel_labels_to_face_labels(mesh, labels)

        parts = split_mesh_by_labels(mesh, face_labels, min_faces=min_segment_faces)

        out_paths = []
        for i, part in enumerate(parts):
            # Simplify if over max_faces
            if hasattr(part, 'faces') and len(part.faces) > max_faces:
                part = part.simplify_quadric_decimation(max_faces)

            path = os.path.join(out_dir, f"part_{i:02d}.glb")
            part.export(path)
            out_paths.append(path)
            log.info(f"SegviGen: exported part {i} → {path}")

        result = "\n".join(out_paths)
        log.info(f"SegviGen: exported {len(parts)} parts to {out_dir}")
        return (result,)


def _voxel_labels_to_face_labels(mesh, voxel_labels) -> "np.ndarray":
    """
    Convert per-voxel label array [R,R,R] to per-face label array [F].

    Strategy: use face centroid positions, normalize to voxel grid coords,
    lookup the label at each centroid voxel.
    """
    import numpy as np

    if voxel_labels is None or mesh is None:
        return np.zeros(0, dtype=np.int32)

    if not hasattr(mesh, 'faces'):
        import trimesh
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

    n_faces = len(mesh.faces)
    if n_faces == 0:
        return np.zeros(0, dtype=np.int32)

    R = voxel_labels.shape[0]
    centroids = mesh.triangles_center  # [F, 3]

    bounds_min = mesh.bounds[0]
    bounds_max = mesh.bounds[1]
    extent = np.maximum(bounds_max - bounds_min, 1e-8)
    norm = (centroids - bounds_min) / extent  # [F, 3] in [0, 1]
    idx = (norm * (R - 1)).round().astype(np.int32).clip(0, R - 1)

    face_labels = voxel_labels[idx[:, 0], idx[:, 1], idx[:, 2]]
    return face_labels.astype(np.int32)
