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

    # Subsample BEFORE coloring: _apply_label_colors assigns one color per face,
    # so we must reduce face count first — coloring a decimated mesh would
    # produce wrong colors since face indices change.
    # This also prevents matplotlib from hanging on 1M+ face meshes.
    # We use random face subsampling (no extra deps) rather than quadric
    # decimation (requires fast_simplification / pyfqmr which may be absent).
    if hasattr(mesh, 'faces') and len(mesh.faces) > _MAX_PREVIEW_FACES:
        import trimesh as _trimesh
        orig_faces = len(mesh.faces)
        try:
            keep = np.sort(
                np.random.choice(orig_faces, _MAX_PREVIEW_FACES, replace=False)
            )
            mesh = _trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces[keep],
                process=False,
            )
            log.info(
                f"SegviGen renderer: subsampled {orig_faces} → {len(mesh.faces)} faces "
                f"for preview (limit={_MAX_PREVIEW_FACES})"
            )
        except Exception as e:
            log.warning(f"SegviGen renderer: face subsampling failed ({e}), using full mesh")

    # Apply per-face segment colors to mesh
    colored_mesh = _apply_label_colors(mesh, labels)

    # Try renderers in order: nvdiffrast → trimesh/pyglet → matplotlib → PIL painter.
    # Each is a hard fallback; the PIL painter requires only Pillow + numpy (always
    # available in ComfyUI) so the chain always terminates with valid images.
    frames = None
    try:
        frames = _render_nvdiffrast(colored_mesh, num_views, resolution)
    except (ImportError, NotImplementedError):
        pass

    if frames is None:
        try:
            frames = _render_trimesh_software(colored_mesh, num_views, resolution)
        except Exception as e:
            log.warning("SegviGen: trimesh renderer failed (%s), using PIL painter", e)

    if frames is None:
        frames = _render_pil_painter(colored_mesh, num_views, resolution)

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


def _render_trimesh_software(mesh, num_views: int, resolution: int):
    """
    Software renderer using trimesh's scene.save_image (requires pyglet).
    Returns a list of PIL Images, or raises an exception if pyglet is absent.
    """
    import trimesh
    import math
    import io
    from PIL import Image

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
        frames.append(Image.open(io.BytesIO(png)).convert("RGB"))

    log.info("SegviGen: trimesh/pyglet renderer succeeded")
    return frames


def _render_pil_painter(mesh, num_views: int, resolution: int) -> list:
    """
    Pure PIL + numpy painter's-algorithm renderer.

    Rotates the mesh around the Y-axis, projects orthographically,
    sorts faces back-to-front by centroid Z, and draws filled triangles
    with PIL.ImageDraw.  Requires only Pillow + numpy — always available
    in ComfyUI.
    """
    import math
    from PIL import Image, ImageDraw

    verts = mesh.vertices.astype(np.float64)
    faces = mesh.faces

    # Per-face colors: fall back to gray if visual has no face_colors
    if hasattr(mesh.visual, "face_colors") and mesh.visual.face_colors is not None:
        fc_rgba = np.asarray(mesh.visual.face_colors, dtype=np.uint8)
        if fc_rgba.shape[1] >= 3:
            face_rgb = fc_rgba[:, :3]
        else:
            face_rgb = np.full((len(faces), 3), 180, dtype=np.uint8)
    else:
        face_rgb = np.full((len(faces), 3), 180, dtype=np.uint8)

    # Normalise mesh to [-1, 1] unit cube centred at origin
    bounds_min = verts.min(axis=0)
    bounds_max = verts.max(axis=0)
    center = (bounds_min + bounds_max) * 0.5
    scale = float((bounds_max - bounds_min).max()) * 0.5
    if scale < 1e-8:
        scale = 1.0
    verts = (verts - center) / scale   # in [-1, 1]

    # Fixed elevation angle  (15°)
    elev_rad = math.radians(15.0)
    cos_e, sin_e = math.cos(elev_rad), math.sin(elev_rad)
    R_elev = np.array([
        [1,      0,     0],
        [0,  cos_e, -sin_e],
        [0,  sin_e,  cos_e],
    ])

    frames = []
    half = resolution / 2.0
    margin = 0.85  # fraction of half-resolution used by the mesh

    for i in range(num_views):
        azim_rad = 2.0 * math.pi * i / num_views
        cos_a, sin_a = math.cos(azim_rad), math.sin(azim_rad)

        # Rotation around Y-axis (azimuth)
        R_azim = np.array([
            [ cos_a, 0, sin_a],
            [     0, 1,     0],
            [-sin_a, 0, cos_a],
        ])
        R = R_elev @ R_azim
        v = verts @ R.T  # [V, 3] in view space

        # Orthographic projection: x,y → screen; z is depth
        # Scale so ±1 world-unit → ±half*margin pixels
        px = (v[:, 0] * half * margin + half).astype(np.float32)
        py = (-v[:, 1] * half * margin + half).astype(np.float32)  # flip Y

        # Face centroid depths for painter's sort (farthest first)
        fv_z = v[faces, 2]           # [F, 3]
        depth = fv_z.mean(axis=1)
        order = np.argsort(depth)[::-1]

        img = Image.new("RGB", (resolution, resolution), (240, 240, 240))
        draw = ImageDraw.Draw(img)

        for fi in order:
            tri = faces[fi]
            pts = [
                (float(px[tri[0]]), float(py[tri[0]])),
                (float(px[tri[1]]), float(py[tri[1]])),
                (float(px[tri[2]]), float(py[tri[2]])),
            ]
            color = tuple(int(c) for c in face_rgb[fi])
            draw.polygon(pts, fill=color)

        frames.append(img)

    log.info("SegviGen: PIL painter renderer succeeded (%d views, %dpx)", num_views, resolution)
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
