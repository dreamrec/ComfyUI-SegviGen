"""
SegviGenMeshPicker: interactive 3D mesh click-to-pick voxel coordinates.

Workflow:
  1. Connect SEGVIGEN_SLAT and (optionally) TRIMESH or choose a glb_file.
  2. Queue the workflow once — this saves/registers the mesh for the picker.
  3. Click '🎯 Open 3D Picker' on the node.
  4. Click points on the mesh surface; confirm to store the coordinates.
  5. Queue again — the stored voxel coords feed SegviGenInteractiveSampler.
"""
import logging
import os
import json as _json

log = logging.getLogger("segvigen")

_MESH_EXTS = (".glb", ".obj", ".ply")


class SegviGenMeshPicker:
    """Interactive 3D mesh point picker for interactive part segmentation."""

    CATEGORY = "SegviGen"
    FUNCTION = "pick"
    RETURN_TYPES = ("SEGVIGEN_POINTS",)
    RETURN_NAMES = ("points",)
    OUTPUT_NODE = True  # makes onExecuted fire so JS receives the mesh filename

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "slat": ("SEGVIGEN_SLAT",),
                # Stores [[x,y,z], ...] JSON — managed by the JS extension.
                "picked_points_json": (
                    "STRING",
                    {
                        "default": "[]",
                        "multiline": False,
                        "tooltip": (
                            "JSON array of voxel [x,y,z] coordinates. "
                            "Use the '🎯 Open 3D Picker' button to set these."
                        ),
                    },
                ),
            },
            "optional": {
                # Live TRIMESH from TRELLIS2 pipeline takes priority over glb_file.
                "trimesh": ("TRIMESH",),
                # Fallback: type a path like 'output/mesh.glb' or 'input/mesh.glb'.
                "glb_file": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Path to mesh file, e.g. 'output/mesh.glb'. "
                        "Used when no TRIMESH input is connected."
                    ),
                }),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def pick(
        self,
        slat,
        picked_points_json="[]",
        trimesh=None,
        glb_file="",
        unique_id=None,
    ):
        import folder_paths

        voxel_resolution = (slat.get("voxel") or {}).get("resolution", 64)

        mesh_filename = ""
        mesh_type     = "output"   # for /view?type=... URL parameter

        if trimesh is not None:
            # Export the live TRIMESH to a temp GLB in the output directory.
            out_dir = folder_paths.get_output_directory()
            mesh_filename = f"segvigen_picker_{unique_id}.glb"
            glb_path = os.path.join(out_dir, mesh_filename)
            try:
                trimesh.export(glb_path)
                log.info(f"SegviGen picker: saved trimesh → {glb_path}")
            except Exception as exc:
                log.warning(f"SegviGen picker: could not save trimesh: {exc}")
                mesh_filename = ""
            mesh_type = "output"

        elif glb_file and not glb_file.startswith("("):
            # Resolve a file from the output/input prefix the dropdown uses.
            if glb_file.startswith("output/"):
                base = folder_paths.get_output_directory()
                mesh_filename = glb_file[len("output/"):]
                mesh_type = "output"
            elif glb_file.startswith("input/"):
                base = folder_paths.get_input_directory()
                mesh_filename = glb_file[len("input/"):]
                mesh_type = "input"
            else:
                # Raw absolute path — copy to output so it's HTTP-accessible.
                base = ""
                mesh_filename = ""

            if mesh_filename:
                resolved = os.path.join(base, mesh_filename)
                if not os.path.isfile(resolved):
                    log.warning(f"SegviGen picker: mesh file not found: {resolved}")
                    mesh_filename = ""
                else:
                    log.info(f"SegviGen picker: using file {resolved}")

        # Parse the stored voxel coordinates.
        try:
            points = _json.loads(picked_points_json)
        except Exception:
            points = []
        if not isinstance(points, list):
            points = []
        clean = []
        for pt in points:
            if isinstance(pt, (list, tuple)) and len(pt) >= 3:
                clean.append([int(pt[0]), int(pt[1]), int(pt[2])])
        points = clean

        log.info(
            f"SegviGen picker: {len(points)} point(s), "
            f"resolution={voxel_resolution}, mesh={mesh_filename!r}"
        )

        # Update the BFS preview cache so the picker UI can highlight voxel
        # components when the user hovers/clicks in the 3D viewer.
        # Must happen here (main process, every run) — the sampler only runs
        # after points are already selected, which is too late for first-run.
        if unique_id is not None and slat.get("voxel") is not None:
            try:
                from core.preview_cache import store as _cache_store
                _cache_store(
                    str(unique_id),
                    {"slat": slat, "voxel_resolution": voxel_resolution},
                )
            except Exception as _exc:
                log.warning(f"SegviGen picker: preview cache update failed: {_exc}")

        return {
            "ui": {
                "mesh_filename":   [mesh_filename],
                "mesh_type":       [mesh_type],
                "voxel_resolution":[voxel_resolution],
                "picked_count":    [len(points)],
            },
            "result": (points,),
        }
