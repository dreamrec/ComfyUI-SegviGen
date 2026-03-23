"""
SegviGen voxel pipeline nodes:
  - SegviGenGLBtoVoxel: mesh → SEGVIGEN_VOXEL
  - SegviGenVoxelEncode: SEGVIGEN_VOXEL → SEGVIGEN_SLAT
"""
import logging

from .helpers import check_interrupt

log = logging.getLogger("segvigen")

VOXEL_RESOLUTIONS = [32, 64, 128]


class SegviGenGLBtoVoxel:
    """Convert a GLB mesh to SegviGen's internal voxel representation."""

    CATEGORY = "SegviGen"
    FUNCTION = "convert"
    RETURN_TYPES = ("SEGVIGEN_VOXEL",)
    RETURN_NAMES = ("voxel",)

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        import os

        # Scan output/ then input/ for mesh files, most-recently-named first.
        _MESH_EXTS = (".glb", ".obj", ".ply")
        choices = []
        for base, tag in [
            (folder_paths.get_output_directory(), "output"),
            (folder_paths.get_input_directory(), "input"),
        ]:
            if os.path.isdir(base):
                files = sorted(
                    (f for f in os.listdir(base)
                     if os.path.isfile(os.path.join(base, f))
                     and f.lower().endswith(_MESH_EXTS)),
                    reverse=True,  # newest timestamps first (TRELLIS2 naming)
                )
                choices.extend(f"{tag}/{f}" for f in files)

        if not choices:
            choices = ["(no mesh files found)"]

        return {
            "required": {},
            "optional": {
                "trimesh": ("TRIMESH", {
                    "tooltip": "Mesh directly from a TRELLIS2 node — skips glb_file",
                }),
                "glb_file": (choices, {
                    "tooltip": "GLB/OBJ/PLY file from output/ or input/ folder",
                }),
                "voxel_resolution": (VOXEL_RESOLUTIONS, {
                    "default": 64,
                    "tooltip": "Voxel grid side length. Higher = more detail, more VRAM.",
                }),
                "simplify_faces": ("INT", {
                    "default": 100_000, "min": 10_000, "max": 500_000, "step": 10_000,
                    "tooltip": "Simplify mesh to this face count before voxelization.",
                }),
            },
        }

    def convert(self, trimesh=None, glb_file="", voxel_resolution=64, simplify_faces=100_000):
        import folder_paths
        import os
        from core.voxel import mesh_to_voxel_grid

        if trimesh is not None:
            source = trimesh
        else:
            if not glb_file or glb_file.startswith("("):
                raise ValueError(
                    "SegviGenGLBtoVoxel: connect a TRIMESH input or select a mesh file "
                    "from the glb_file dropdown."
                )
            # Resolve prefixed path (e.g. "output/trellis2_xxx.glb")
            if glb_file.startswith("output/"):
                source = os.path.join(folder_paths.get_output_directory(), glb_file[7:])
            elif glb_file.startswith("input/"):
                source = os.path.join(folder_paths.get_input_directory(), glb_file[6:])
            else:
                source = glb_file  # raw absolute path fallback

            if not os.path.isfile(source):
                raise FileNotFoundError(
                    f"SegviGenGLBtoVoxel: mesh file not found: {source}"
                )

        log.info(f"SegviGen: converting mesh to {voxel_resolution}³ voxel grid")
        voxel = mesh_to_voxel_grid(source, resolution=voxel_resolution,
                                    simplify_faces=simplify_faces)
        return (voxel,)


class SegviGenVoxelEncode:
    """
    Encode a voxel grid into a SLAT latent using SegviGen's bundled SLAT encoder.

    IMPORTANT: This uses SegviGen's own encoder model (downloaded via install.py),
    NOT a TRELLIS2 stage function. stages.py does not expose voxel encoding.
    The SegviGen checkpoint bundle includes a shape SLAT encoder separate from
    TRELLIS2's shape generation pipeline.
    """

    CATEGORY = "SegviGen"
    FUNCTION = "encode"
    RETURN_TYPES = ("SEGVIGEN_SLAT",)
    RETURN_NAMES = ("slat",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_config": ("TRELLIS2_MODEL_CONFIG", {
                    "tooltip": "Config from Load TRELLIS2 Models node",
                }),
                "voxel": ("SEGVIGEN_VOXEL",),
            },
        }

    def encode(self, model_config: dict, voxel: dict):
        import comfy.model_management as mm
        from core.pipeline import encode_voxel_to_slat, get_encoder_patcher
        import folder_paths, os

        check_interrupt()

        models_dir = os.path.join(folder_paths.models_dir, "segvigen")
        patcher = get_encoder_patcher(model_config, models_dir)
        mm.load_models_gpu([patcher])

        try:
            # Pass the numpy grid (not the full dict) to the encoder
            grid = voxel["grid"]
            latent = encode_voxel_to_slat(patcher.model, grid, model_config)
        finally:
            mm.soft_empty_cache()

        return ({"latent": latent, "voxel": voxel},)
