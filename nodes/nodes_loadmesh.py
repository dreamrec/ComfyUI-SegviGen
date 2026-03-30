"""
SegviGenLoadMesh: load a GLB/OBJ/PLY file into a TRIMESH object.

Shows a file-browser/upload widget (same mechanism as ComfyUI's Load3D node)
so the user can click to upload or select any 3D file — no manual path entry.
"""
import logging
import os

log = logging.getLogger("segvigen")

_MESH_EXTS = (".glb", ".gltf", ".obj", ".ply", ".fbx", ".stl")


class SegviGenLoadMesh:
    """
    Load a GLB/OBJ/PLY mesh file and output it as a TRIMESH object.

    • Files dropped into ComfyUI/input/3d/ appear in the dropdown.
    • The upload button (📎) lets you upload directly from your computer.
    • Recent TRELLIS2 output GLBs also appear prefixed with 'output/'.
    """

    CATEGORY  = "SegviGen"
    FUNCTION  = "load"
    RETURN_TYPES = ("TRIMESH", "STRING")
    RETURN_NAMES = ("trimesh", "model_file")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        files = []

        # 1. input/3d/ — the standard folder used by ComfyUI's own Load3D node.
        #    Creates the directory on first run so users know where to drop files.
        input_3d = os.path.join(folder_paths.get_input_directory(), "3d")
        try:
            os.makedirs(input_3d, exist_ok=True)
            for f in sorted(os.listdir(input_3d)):
                if os.path.splitext(f)[1].lower() in _MESH_EXTS:
                    files.append(f"3d/{f}")
        except Exception:
            pass

        # 2. Plain input/ root (for files not in a subdirectory).
        try:
            inp = folder_paths.get_input_directory()
            for f in sorted(os.listdir(inp)):
                full = os.path.join(inp, f)
                if os.path.isfile(full) and os.path.splitext(f)[1].lower() in _MESH_EXTS:
                    files.append(f)
        except Exception:
            pass

        # 3. output/ — recent TRELLIS2 / SegviGen-exported meshes (newest first).
        try:
            out = folder_paths.get_output_directory()
            out_files = sorted(
                (f for f in os.listdir(out)
                 if os.path.isfile(os.path.join(out, f))
                 and os.path.splitext(f)[1].lower() in _MESH_EXTS),
                reverse=True,
            )
            files.extend(f"output/{f}" for f in out_files[:30])
        except Exception:
            pass

        if not files:
            files = ["(no 3D files — upload one via the button)"]

        return {
            "required": {
                "model_file": (
                    files,
                    {
                        # "file_upload" = IO.UploadType.model in the newer API.
                        # This adds the 📎 upload button next to the dropdown,
                        # letting users browse their filesystem directly.
                        "upload": "file_upload",
                        "tooltip": (
                            "Select an existing mesh or click the upload button "
                            "to browse your filesystem. Files are stored in "
                            "ComfyUI/input/3d/."
                        ),
                    },
                ),
            },
        }

    def load(self, model_file: str):
        import trimesh as _tm
        import folder_paths

        path = self._resolve_path(model_file, folder_paths)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"SegviGenLoadMesh: file not found: {path!r}\n"
                f"  Tried to resolve: {model_file!r}"
            )

        log.info(f"SegviGen: loading mesh from {path}")
        loaded = _tm.load(str(path), force="mesh")
        if isinstance(loaded, _tm.Scene):
            loaded = loaded.dump(concatenate=True)
        if not isinstance(loaded, _tm.Trimesh):
            raise ValueError(
                f"SegviGenLoadMesh: could not load a Trimesh from {path!r} "
                f"(got {type(loaded).__name__})"
            )

        log.info(
            f"SegviGen: mesh loaded — {len(loaded.faces)} faces, "
            f"{len(loaded.vertices)} vertices"
        )

        # Export a GLB copy to output/ so Preview3D can display it
        out_glb = os.path.join(
            folder_paths.get_output_directory(),
            f"segvigen_input_{os.path.splitext(os.path.basename(path))[0]}.glb",
        )
        loaded.export(out_glb, file_type="glb")

        return (loaded, out_glb)

    @staticmethod
    def _resolve_path(model_file: str, folder_paths) -> str:
        """Resolve a model_file string (possibly with output/ prefix) to an
        absolute file path."""
        if model_file.startswith("output/"):
            return os.path.join(
                folder_paths.get_output_directory(),
                model_file[len("output/"):]
            )
        # Use ComfyUI's own resolver (handles 3d/ subfolders, annotations, etc.)
        try:
            resolved = folder_paths.get_annotated_filepath(model_file)
            if resolved and os.path.isfile(resolved):
                return resolved
        except Exception:
            pass
        # Fallback: treat as relative to input/
        return os.path.join(folder_paths.get_input_directory(), model_file)
