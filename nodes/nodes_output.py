"""
SegviGen output nodes:
  - SegviGenRenderPreview: 2D colored segment preview images
  - SegviGenExportParts (Splitter): split mesh into named parts, export combined + individual GLBs
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
    """Split segmented mesh into named parts and export as GLB files.

    Outputs:
      - combined_file: single GLB with all parts as named sub-objects
        (opens in Blender/Unity/etc. as separate selectable objects)
      - individual_parts: newline-separated paths to individual part GLBs
    """

    CATEGORY = "SegviGen"
    FUNCTION = "export"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("combined_file", "individual_parts")

    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seg_result": ("SEGVIGEN_SEG_RESULT",),
            },
            "optional": {
                "max_faces": ("INT", {
                    "default": 100_000, "min": 10_000, "max": 500_000, "step": 10_000,
                }),
                "min_segment_faces": ("INT", {
                    "default": 50, "min": 1, "max": 10_000, "step": 10,
                    "tooltip": "Discard segments smaller than this face count.",
                }),
                "filename_prefix": ("STRING", {"default": "segvigen"}),
            },
        }

    def export(
        self,
        seg_result: dict,
        max_faces: int = 100_000,
        min_segment_faces: int = 50,
        filename_prefix: str = "segvigen",
    ):
        import numpy as np
        import trimesh
        import folder_paths
        from core.split import split_mesh_by_labels
        from core.renderer import SEGMENT_COLORS, _voxel_labels_to_face_labels

        check_interrupt()

        mesh = seg_result.get("mesh")
        labels = seg_result.get("labels")

        if mesh is None:
            log.warning("SegviGen export: no mesh in seg_result, skipping export")
            return ("", "")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(folder_paths.output_directory, "segvigen", timestamp)
        os.makedirs(out_dir, exist_ok=True)

        # Convert per-voxel labels to per-face labels
        face_labels = _voxel_labels_to_face_labels(mesh, labels)
        parts = split_mesh_by_labels(mesh, face_labels, min_faces=min_segment_faces)

        # ── 1. Combined GLB: all parts as named sub-objects in one file ──
        scene = trimesh.Scene()
        for i, part in enumerate(parts):
            if hasattr(part, 'faces') and len(part.faces) > max_faces:
                part = _simplify(part, max_faces)

            # Color each part with its segment color
            n = len(part.faces)
            fc = np.full((n, 4), 255, dtype=np.uint8)
            fc[:, :3] = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
            part.visual = trimesh.visual.ColorVisuals(face_colors=fc)

            name = f"part_{i:02d}"
            scene.add_geometry(part, node_name=name, geom_name=name)

        combined_path = os.path.join(out_dir, f"{filename_prefix}_parts.glb")
        scene.export(combined_path, file_type="glb")
        log.info(f"SegviGen: combined GLB with {len(parts)} named parts → {combined_path}")

        # ── 2. Individual GLB files (one per part) ───────────────────────
        individual_paths = []
        for i, part in enumerate(parts):
            if hasattr(part, 'faces') and len(part.faces) > max_faces:
                part = _simplify(part, max_faces)
            path = os.path.join(out_dir, f"part_{i:02d}.glb")
            part.export(path, file_type="glb")
            individual_paths.append(path)
            log.info(f"SegviGen: part {i} ({len(part.faces)} faces) → {path}")

        return (combined_path, "\n".join(individual_paths))


def _simplify(mesh, target_faces: int):
    """Simplify mesh to target face count."""
    if len(mesh.faces) <= target_faces:
        return mesh
    return mesh.simplify_quadric_decimation(face_count=target_faces)
