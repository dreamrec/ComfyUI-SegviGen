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

# Segment colors matching the picker's MARKER_COLORS so the 2D preview
# uses the same color for each component as the 3D picker dots.
# Source: web/picker.html MARKER_COLORS array (hex → RGB).
SEGMENT_COLORS = np.array([
    [0xff, 0x44, 0x55],  # P1 red        (#ff4455)
    [0xff, 0x8c, 0x00],  # P2 orange     (#ff8c00)
    [0xff, 0xdd, 0x00],  # P3 yellow     (#ffdd00)
    [0x44, 0xee, 0x66],  # P4 green      (#44ee66)
    [0x22, 0xcc, 0xee],  # P5 cyan       (#22ccee)
    [0x88, 0x55, 0xff],  # P6 purple     (#8855ff)
    [0xff, 0x55, 0xcc],  # P7 pink       (#ff55cc)
    [0x00, 0xff, 0xaa],  # P8 mint       (#00ffaa)
    [0xff, 0x99, 0x33],  # P9 amber      (#ff9933)
    [0xaa, 0xff, 0xdd],  # P10 light grn (#aaffdd)
], dtype=np.uint8)


_MAX_PREVIEW_FACES = 30_000  # matplotlib hangs above ~50k; keep well under that


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

    # Decimate BEFORE coloring: _apply_label_colors assigns one color per face,
    # so we must decimate first — decimation merges faces and would discard colors.
    # This also prevents matplotlib from hanging on 1M+ face meshes.
    if hasattr(mesh, 'faces') and len(mesh.faces) > _MAX_PREVIEW_FACES:
        orig_faces = len(mesh.faces)
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=_MAX_PREVIEW_FACES)
            log.info(
                f"SegviGen renderer: decimated {orig_faces} → {len(mesh.faces)} faces "
                f"for preview (limit={_MAX_PREVIEW_FACES})"
            )
        except Exception as e:
            log.warning(f"SegviGen renderer: decimation failed ({e}), using full mesh")

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
            face_labels = _voxel_labels_to_face_labels(mesh, labels)
            # Vectorized: color all faces with label > 0
            fg_mask = face_labels > 0
            if fg_mask.any():
                # Labels are 1-based (BFS assigns 1,2,3...; 0=background).
                color_idx = (face_labels[fg_mask] - 1) % len(SEGMENT_COLORS)
                face_colors[fg_mask, :3] = SEGMENT_COLORS[color_idx]
            log.info(f"SegviGen renderer: {int(fg_mask.sum())}/{n_faces} faces colored "
                     f"({len(np.unique(face_labels[fg_mask]))} segments)")
        except Exception as e:
            log.warning("SegviGen renderer: label-to-face mapping failed (%s), using gray", e)

    colored = mesh.copy()
    colored.visual = trimesh.visual.ColorVisuals(face_colors=face_colors)
    return colored


def _render_trimesh_software(mesh, num_views: int, resolution: int) -> list:
    """Software renderer fallback using trimesh's scene.save_image (pyglet)."""
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
        try:
            scene.set_camera(eye, center)
            png = scene.save_image(resolution=(resolution, resolution))
            from PIL import Image
            import io
            frames.append(Image.open(io.BytesIO(png)).convert("RGB"))
        except Exception as e:
            if not frames:
                # First frame failed — pyglet/GL context issue, use matplotlib fallback
                log.warning("SegviGen: trimesh save_image failed (%s), using matplotlib fallback", e)
                return _render_matplotlib_fallback(mesh, num_views, resolution)
            # Later frame failed — fill with last successful frame
            frames.append(frames[-1].copy())

    return frames


def _render_matplotlib_fallback(mesh, num_views: int, resolution: int) -> list:
    """Matplotlib 3D renderer — produces clean square images with correct aspect ratio."""
    import math
    import io
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    verts = mesh.vertices
    faces = mesh.faces
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    # Use uniform extent on all axes so the mesh isn't distorted
    half_extent = float((bounds[1] - bounds[0]).max()) / 2.0 * 1.15

    # Build face polygons with colors
    face_verts = verts[faces]
    fc = None
    if hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None:
        fc = mesh.visual.face_colors[:, :3].astype(np.float32) / 255.0

    dpi = 100
    figsize = resolution / dpi

    frames = []
    for i in range(num_views):
        angle_deg = 360.0 * i / num_views
        fig = plt.figure(figsize=(figsize, figsize), dpi=dpi, facecolor="white")
        # Fill the entire figure with the 3D axes — no margins
        ax = fig.add_axes([0, 0, 1, 1], projection="3d")
        ax.set_facecolor("white")
        poly = Poly3DCollection(face_verts, linewidths=0.05, edgecolors=(0.4, 0.4, 0.4, 0.15))
        if fc is not None:
            poly.set_facecolor(fc)
        else:
            poly.set_facecolor((0.7, 0.7, 0.7))
        ax.add_collection3d(poly)
        # Uniform axis limits centered on the mesh — prevents squashing
        ax.set_xlim(center[0] - half_extent, center[0] + half_extent)
        ax.set_ylim(center[1] - half_extent, center[1] + half_extent)
        ax.set_zlim(center[2] - half_extent, center[2] + half_extent)
        # Force equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=20, azim=angle_deg)
        ax.axis("off")
        buf = io.BytesIO()
        # Use pad_inches=0 but NOT bbox_inches="tight" — this preserves
        # the exact figsize we specified instead of cropping to content
        fig.savefig(buf, format="png", dpi=dpi, pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        # Should already be resolution×resolution, but ensure it
        if img.size != (resolution, resolution):
            img = img.resize((resolution, resolution), Image.LANCZOS)
        frames.append(img)

    return frames


def _voxel_labels_to_face_labels(mesh, voxel_labels) -> "np.ndarray":
    """
    Convert per-voxel label array [R,R,R] to per-face label array [F].

    Strategy:
    1. First try direct grid lookup (fast path).
    2. For faces that land on empty voxels (label=0), use KD-tree nearest-neighbor
       search against occupied voxels to find the closest labeled voxel.
       This handles the sparsity of the voxel grid — the 64³ grid has ~30k occupied
       voxels out of 262k total, so many face centroids land in empty space.

    Uses the same uniform-scale normalization as voxel.py's _normalize_to_unit_cube:
    center the mesh, scale by 1/max_extent so the longest axis fits in [-0.5, 0.5].
    """
    import trimesh as _trimesh

    if voxel_labels is None or mesh is None:
        return np.zeros(0, dtype=np.int32)

    if not hasattr(mesh, 'faces'):
        if isinstance(mesh, _trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

    n_faces = len(mesh.faces)
    if n_faces == 0:
        return np.zeros(0, dtype=np.int32)

    R = voxel_labels.shape[0]
    centroids = mesh.triangles_center  # [F, 3]

    bounds_min = mesh.bounds[0]
    bounds_max = mesh.bounds[1]
    center     = (bounds_min + bounds_max) / 2.0
    max_extent = float((bounds_max - bounds_min).max())
    if max_extent < 1e-8:
        return np.zeros(n_faces, dtype=np.int32)
    norm = (centroids - center) / max_extent   # [F, 3] in approx [-0.5, 0.5]

    # Convert normalized coords to voxel grid coordinates (continuous)
    voxel_coords = (norm + 0.5) * R  # [F, 3] in [0, R]
    idx = np.round(voxel_coords).astype(np.int32).clip(0, R - 1)

    # Fast path: direct grid lookup
    face_labels = voxel_labels[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.int32)

    # Fix unlabeled faces using KD-tree nearest-neighbor against occupied voxels
    unlabeled = face_labels == 0
    n_unlabeled = int(unlabeled.sum())
    if n_unlabeled > 0:
        # Build KD-tree from occupied (labeled) voxel positions
        occupied_mask = voxel_labels > 0
        occupied_ijk = np.argwhere(occupied_mask)  # [K, 3]
        if len(occupied_ijk) > 0:
            from scipy.spatial import cKDTree
            tree = cKDTree(occupied_ijk.astype(np.float64))
            # Query for each unlabeled face centroid (in voxel coords)
            query_pts = voxel_coords[unlabeled]  # [U, 3]
            _, nn_idx = tree.query(query_pts, k=1)
            nn_ijk = occupied_ijk[nn_idx]
            face_labels[unlabeled] = voxel_labels[
                nn_ijk[:, 0], nn_ijk[:, 1], nn_ijk[:, 2]
            ].astype(np.int32)
            log.info(f"SegviGen: KD-tree fixed {n_unlabeled} unlabeled faces "
                     f"(out of {n_faces} total)")

    return face_labels


def _render_nvdiffrast(mesh, num_views: int, resolution: int) -> list:
    """GPU-accelerated rendering via nvdiffrast."""
    import nvdiffrast.torch as dr
    # Placeholder — nvdiffrast requires rasterization setup with
    # vertex/index buffers, MVP matrices, and a RasterizeGLContext.
    # Implement following TRELLIS2's nvdiffrec_render usage pattern in stages.py.
    raise NotImplementedError("nvdiffrast rendering — implement following TRELLIS2 pattern")


def _placeholder_image(num_views: int, resolution: int) -> torch.Tensor:
    return torch.zeros(num_views, resolution, resolution, 3, dtype=torch.float32)
