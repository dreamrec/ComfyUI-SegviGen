"""
Palette-based GLB mesh splitter for SegviGen.

split_mesh_by_labels: given a mesh and a per-face integer label array,
return a list of sub-meshes (one per unique label), filtered by min_faces.

Cleaned from Aero-Ex's split.py — removed hardcoded path constants,
simplified API, same core topology-correction algorithm.
"""
import logging
import numpy as np

log = logging.getLogger("segvigen")


def split_mesh_by_labels(
    mesh,
    face_labels: np.ndarray,
    min_faces: int = 50,
) -> list:
    """
    Split a mesh into sub-meshes based on per-face integer labels.

    Args:
        mesh: trimesh.Trimesh — the unsplit mesh
        face_labels: int32 array of length len(mesh.faces) — segment index per face.
                     -1 labels are ignored.
        min_faces: discard segments with fewer faces than this threshold

    Returns:
        List of trimesh.Trimesh objects, one per valid segment.

    Raises:
        ValueError: if no valid segments remain after filtering.
    """
    import trimesh

    unique_labels = np.unique(face_labels)
    unique_labels = unique_labels[unique_labels >= 0]

    parts = []
    for label in unique_labels:
        face_mask = face_labels == label
        face_indices = np.where(face_mask)[0]

        if len(face_indices) < min_faces:
            log.debug(f"SegviGen split: label {label} has {len(face_indices)} faces < {min_faces}, skipping")
            continue

        submesh = mesh.submesh([face_indices], append=True)
        if submesh is not None and len(submesh.faces) > 0:
            parts.append(submesh)

    if not parts:
        raise ValueError(
            f"No valid segments found (all segments below min_faces={min_faces}). "
            "Try lowering min_segment_faces or adjusting segmentation parameters."
        )

    log.info(f"SegviGen split: {len(parts)} parts extracted from {len(unique_labels)} labels")
    return parts
