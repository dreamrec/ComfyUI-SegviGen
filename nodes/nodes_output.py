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
        # Individual parts go in a timestamped subfolder (stays tidy).
        parts_dir = os.path.join(folder_paths.output_directory, "segvigen", timestamp)
        os.makedirs(parts_dir, exist_ok=True)

        # ── No points yet: export the full unsegmented mesh as a preview ──
        # labels=None means the sampler did a first-run passthrough (no click
        # points have been set).  Export the whole mesh so Preview3D shows the
        # new object straight away, giving the user something to click on in
        # the 3D picker without the workflow crashing.
        if labels is None:
            whole_mesh = mesh
            if hasattr(mesh, 'faces'):
                n_whole = len(mesh.faces)
            elif hasattr(mesh, 'dump'):
                whole_mesh = mesh.dump(concatenate=True)
                n_whole = len(whole_mesh.faces)
            else:
                n_whole = 0

            if n_whole > max_faces:
                whole_mesh = _simplify(whole_mesh, max_faces)

            combined_filename = f"{filename_prefix}_preview_{timestamp}.glb"
            combined_path = os.path.join(folder_paths.output_directory, combined_filename)
            whole_mesh.export(combined_path, file_type="glb")
            log.info(
                f"SegviGen: no points yet — exported full mesh preview → {combined_path}. "
                "Click '🎯 Open 3D Picker', select part(s), then run again."
            )
            return (combined_filename, "")

        # Convert per-voxel labels to per-face labels
        face_labels = _voxel_labels_to_face_labels(mesh, labels)
        parts = split_mesh_by_labels(mesh, face_labels, min_faces=min_segment_faces)

        if not parts:
            # Still export the whole mesh so Preview3D shows something.
            # This happens when: all predicted segments are below min_segment_faces,
            # or the label-to-face mapping produced no foreground labels.
            log.info(
                "SegviGen: no valid segments — exporting full mesh as fallback preview. "
                "Try lowering min_segment_faces, or click more points and run again."
            )
            fallback = mesh
            if not hasattr(fallback, 'faces') and hasattr(fallback, 'dump'):
                fallback = fallback.dump(concatenate=True)
            if hasattr(fallback, 'faces') and len(fallback.faces) > max_faces:
                fallback = _simplify(fallback, max_faces)
            fallback_filename = f"{filename_prefix}_fallback_{timestamp}.glb"
            fallback_path = os.path.join(folder_paths.output_directory, fallback_filename)
            fallback.export(fallback_path, file_type="glb")
            return (fallback_filename, "")

        # ── 1. Combined GLB: all parts as named sub-objects in one file ──
        # Saved to the OUTPUT ROOT (no subfolder) so ComfyUI's Preview3D node
        # can load it — Preview3D.execute passes the filename string directly to
        # PreviewUI3D which the frontend resolves via /view?filename=X&type=output.
        # An absolute or subfolder path would silently fail to load.
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

        combined_filename = f"{filename_prefix}_parts_{timestamp}.glb"
        combined_path = os.path.join(folder_paths.output_directory, combined_filename)
        scene.export(combined_path, file_type="glb")
        log.info(f"SegviGen: combined GLB ({len(parts)} parts) → {combined_path}")

        # ── 2. Individual GLB files (one per part, in timestamped subfolder) ─
        individual_paths = []
        for i, part in enumerate(parts):
            if hasattr(part, 'faces') and len(part.faces) > max_faces:
                part = _simplify(part, max_faces)
            path = os.path.join(parts_dir, f"part_{i:02d}.glb")
            part.export(path, file_type="glb")
            individual_paths.append(path)
            log.info(f"SegviGen: part {i} ({len(part.faces)} faces) → {path}")

        # Return just the filename (no path) for Preview3D compatibility.
        return (combined_filename, "\n".join(individual_paths))


def _simplify(mesh, target_faces: int):
    """
    Reduce mesh to at most target_faces using random face subsampling.

    We avoid simplify_quadric_decimation (requires fast_simplification / pyfqmr,
    NOT installed in the ComfyUI pixi env).  Random subsampling is dependency-free,
    instant, and sufficient for preview/export where exact topology isn't critical.
    """
    import numpy as np
    import trimesh as _trimesh

    if not hasattr(mesh, 'faces') or len(mesh.faces) <= target_faces:
        return mesh
    orig_faces = len(mesh.faces)
    keep = np.sort(np.random.choice(orig_faces, target_faces, replace=False))
    result = _trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces[keep],
        process=False,
    )
    log.debug(f"SegviGen export: subsampled {orig_faces} → {len(result.faces)} faces")
    return result
