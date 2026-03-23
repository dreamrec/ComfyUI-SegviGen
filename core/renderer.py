"""
Colored segmentation preview renderer using nvdiffrast.

render_segmentation_preview:
  Given a trimesh.Scene or labeled mesh, render num_views images
  with each segment in a distinct color.

nvdiffrast uses a per-process GL/CUDA context created on first use
(not a ComfyUI ModelPatcher — no load_models_gpu needed).

Note: nvdiffrast is available in TRELLIS2's isolated env. If it fails
to import, fall back to trimesh's built-in software renderer.
"""
import logging
import numpy as np
import torch

log = logging.getLogger("segvigen")

# Distinct colors for up to 20 segments (RGB 0-255)
SEGMENT_COLORS = np.array([
    [220, 80,  80],   # red
    [80,  120, 220],  # blue
    [80,  200, 100],  # green
    [220, 180, 60],   # yellow
    [180, 80,  220],  # purple
    [60,  200, 200],  # cyan
    [220, 130, 60],   # orange
    [140, 200, 80],   # lime
    [220, 80,  160],  # pink
    [100, 160, 220],  # sky blue
    [200, 160, 80],   # tan
    [80,  180, 160],  # teal
    [160, 80,  80],   # maroon
    [80,  80,  180],  # navy
    [120, 160, 120],  # sage
    [200, 120, 120],  # salmon
    [120, 200, 160],  # mint
    [160, 120, 200],  # lavender
    [200, 200, 80],   # chartreuse
    [80,  160, 200],  # cerulean
], dtype=np.uint8)


def render_segmentation_preview(
    seg_result: dict,
    num_views: int = 8,
    resolution: int = 512,
) -> torch.Tensor:
    """
    Render colored preview images of the segmentation result.

    Args:
        seg_result: SEGVIGEN_SEG_RESULT dict
        num_views: number of views to render (rotating around the object)
        resolution: render resolution in pixels

    Returns:
        [num_views, H, W, 3] float32 tensor (ComfyUI IMAGE batch format)
    """
    import trimesh
    from PIL import Image

    mesh = seg_result.get("mesh")
    labels = seg_result.get("labels")  # per-voxel [R,R,R] int32

    if mesh is None:
        log.warning("SegviGen renderer: no mesh in seg_result, returning placeholder")
        return _placeholder_image(num_views, resolution)

    # Apply per-face segment colors to mesh
    colored_mesh = _apply_label_colors(mesh, labels)

    # Try nvdiffrast first, fall back to trimesh software renderer.
    # Catch NotImplementedError too — _render_nvdiffrast raises it until implemented.
    try:
        frames = _render_nvdiffrast(colored_mesh, num_views, resolution)
    except (ImportError, NotImplementedError):
        log.warning("SegviGen: nvdiffrast not available / not yet implemented, using trimesh fallback")
        frames = _render_trimesh_software(colored_mesh, num_views, resolution)

    # Stack into [num_views, H, W, 3] float32
    frames_np = np.stack([np.array(f) for f in frames], axis=0)
    return torch.from_numpy(frames_np.astype(np.float32) / 255.0)


def _apply_label_colors(mesh, labels):
    """
    Color mesh faces by segment label.

    Uses _voxel_labels_to_face_labels (centroid-based lookup) to map
    the [R,R,R] voxel label grid to per-face labels.
    """
    import trimesh
    if not isinstance(mesh, trimesh.Trimesh):
        if hasattr(mesh, 'dump'):
            mesh = mesh.dump(concatenate=True)
        else:
            return mesh

    n_faces = len(mesh.faces)
    face_colors = np.full((n_faces, 4), 180, dtype=np.uint8)  # default gray
    face_colors[:, 3] = 255

    if labels is not None and n_faces > 0:
        try:
            from nodes.nodes_output import _voxel_labels_to_face_labels
            face_labels = _voxel_labels_to_face_labels(mesh, labels)
            for i, lbl in enumerate(face_labels):
                face_colors[i, :3] = SEGMENT_COLORS[int(lbl) % len(SEGMENT_COLORS)]
        except Exception:
            log.warning("SegviGen renderer: label-to-face mapping failed, using gray")

    colored = mesh.copy()
    colored.visual = trimesh.visual.ColorVisuals(face_colors=face_colors)
    return colored


def _render_trimesh_software(mesh, num_views: int, resolution: int) -> list:
    """Software renderer fallback using trimesh's scene.save_image."""
    import trimesh
    import math

    scene = trimesh.Scene([mesh])
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    extent = np.linalg.norm(bounds[1] - bounds[0])
    distance = extent * 2.0

    frames = []
    for i in range(num_views):
        angle = 2.0 * math.pi * i / num_views
        eye = center + np.array([
            distance * math.sin(angle),
            distance * 0.3,
            distance * math.cos(angle),
        ])
        scene.set_camera(eye, center)
        png = scene.save_image(resolution=(resolution, resolution))
        from PIL import Image
        import io
        frames.append(Image.open(io.BytesIO(png)).convert("RGB"))

    return frames


def _render_nvdiffrast(mesh, num_views: int, resolution: int) -> list:
    """GPU-accelerated rendering via nvdiffrast."""
    import nvdiffrast.torch as dr
    # Placeholder — nvdiffrast requires rasterization setup with
    # vertex/index buffers, MVP matrices, and a RasterizeGLContext.
    # Implement following TRELLIS2's nvdiffrec_render usage pattern in stages.py.
    raise NotImplementedError("nvdiffrast rendering — implement following TRELLIS2 pattern")


def _placeholder_image(num_views: int, resolution: int) -> torch.Tensor:
    return torch.zeros(num_views, resolution, resolution, 3, dtype=torch.float32)
