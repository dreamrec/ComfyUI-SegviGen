"""
SegviGen voxel pipeline nodes:
  - SegviGenGLBtoVoxel: mesh → SEGVIGEN_VOXEL
  - SegviGenVoxelEncode: TRELLIS2_SHAPE_RESULT + CONDITIONING → SEGVIGEN_SLAT (shape + tex)
  - SegviGenFromShapeResult: TRELLIS2_SHAPE_RESULT → SEGVIGEN_SLAT (shape only)
"""
import logging

from .helpers import check_interrupt

log = logging.getLogger("segvigen")


class SegviGenGLBtoVoxel:
    """Convert a GLB mesh to SegviGen's internal voxel representation.

    Accepts mesh from:
      - TRIMESH input (from SegviGenLoadMesh or TRELLIS2)
      - mesh_path STRING (from ComfyUI's built-in Load3D node)
      - glb_file path string (manual entry)

    Outputs both the voxel grid AND the loaded trimesh, so downstream nodes
    (samplers, preview) can use the mesh without a separate loader.
    """

    CATEGORY = "SegviGen"
    FUNCTION = "convert"
    RETURN_TYPES = ("SEGVIGEN_VOXEL", "TRIMESH")
    RETURN_NAMES = ("voxel", "trimesh")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "trimesh": ("TRIMESH", {
                    "tooltip": "Mesh from SegviGenLoadMesh or TRELLIS2 node.",
                }),
                "mesh_path": ("STRING", {
                    "default": "",
                    "forceInput": True,
                    "tooltip": (
                        "Mesh file path from Load3D's mesh_path output. "
                        "Connect Load3D → this for built-in 3D preview."
                    ),
                }),
                "glb_file": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Manual path to a GLB/OBJ/PLY mesh. Leave empty when "
                        "TRIMESH or mesh_path is connected."
                    ),
                }),
                "voxel_resolution": ("INT", {
                    "default": 64, "min": 1, "max": 0x7FFFFFFF,
                    "tooltip": "Voxel grid side length. Recommended: 32, 64, or 128.",
                }),
                "simplify_faces": ("INT", {
                    "default": 100_000, "min": 10_000, "max": 500_000, "step": 10_000,
                    "tooltip": "Simplify mesh to this face count before voxelization.",
                }),
            },
        }

    def convert(self, trimesh=None, mesh_path="", glb_file="",
                voxel_resolution=64, simplify_faces=100_000):
        import folder_paths
        import os
        import trimesh as _tm
        from core.voxel import mesh_to_voxel_grid

        # Priority: trimesh object > mesh_path from Load3D > glb_file manual path
        mesh_obj = None
        if trimesh is not None:
            source = trimesh
            mesh_obj = trimesh
        elif mesh_path and mesh_path.strip():
            # mesh_path from Load3D — resolve relative to input dir
            path = mesh_path.strip()
            if not os.path.isabs(path):
                path = os.path.join(folder_paths.get_input_directory(), path)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"SegviGenGLBtoVoxel: mesh_path not found: {path}"
                )
            source = path
        elif glb_file and glb_file.strip() and not glb_file.startswith("("):
            path = glb_file.strip()
            if path.startswith("output/"):
                path = os.path.join(folder_paths.get_output_directory(), path[7:])
            elif path.startswith("input/"):
                path = os.path.join(folder_paths.get_input_directory(), path[6:])
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"SegviGenGLBtoVoxel: mesh file not found: {path}"
                )
            source = path
        else:
            raise ValueError(
                "SegviGenGLBtoVoxel: connect a TRIMESH, a Load3D mesh_path, "
                "or enter a glb_file path."
            )

        # Clamp to a safe range
        if voxel_resolution > 256 or voxel_resolution < 16:
            safe = 64
            log.warning(
                f"SegviGen: voxel_resolution={voxel_resolution} is out of the "
                f"recommended 16–256 range; using {safe} instead."
            )
            voxel_resolution = safe

        log.info(f"SegviGen: converting mesh to {voxel_resolution}³ voxel grid")
        voxel = mesh_to_voxel_grid(source, resolution=voxel_resolution,
                                    simplify_faces=simplify_faces)
        voxel["resolution"] = voxel_resolution

        # Load/ensure TRIMESH output for downstream sampler nodes
        if mesh_obj is None:
            loaded = _tm.load(str(source), force="mesh")
            if isinstance(loaded, _tm.Scene):
                loaded = loaded.dump(concatenate=True)
            mesh_obj = loaded
            log.info(f"SegviGen: mesh loaded — {len(mesh_obj.faces)} faces")

        return (voxel, mesh_obj)


class SegviGenVoxelEncode:
    """
    Encode TRELLIS2 shape result + conditioning into a SEGVIGEN_SLAT with real
    shape_slat and tex_slat.

    Uses TRELLIS2's _sample_tex_slat() via stage shim to produce real tex_slat
    from shape_slat + conditioning. Falls back to source="shape_only" if tex
    sampling fails.
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
                "shape_result": ("TRELLIS2_SHAPE_RESULT", {
                    "tooltip": "Connect the shape_result output from TRELLIS2 Image-to-Shape node.",
                }),
                "conditioning": ("TRELLIS2_CONDITIONING", {
                    "tooltip": "Connect the conditioning output from TRELLIS2 conditioning node.",
                }),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1,
                    "tooltip": "Random seed for texture sampling.",
                }),
                "tex_guidance_strength": ("FLOAT", {
                    "default": 7.5, "min": 0.0, "max": 20.0, "step": 0.1,
                    "tooltip": "CFG strength for texture flow model.",
                }),
                "tex_sampling_steps": ("INT", {
                    "default": 12, "min": 1, "max": 50,
                    "tooltip": "Number of sampling steps for texture generation.",
                }),
            },
        }

    def encode(self, model_config: dict, shape_result: dict, conditioning: dict,
               seed: int = 0, tex_guidance_strength: float = 7.5,
               tex_sampling_steps: int = 12):
        import comfy.model_management as mm
        from core.encode import extract_shape_data, sample_tex_slat

        check_interrupt()

        device = str(mm.get_torch_device())
        shape_slat, subs, resolution, pipeline_type = extract_shape_data(
            shape_result, device
        )

        try:
            tex_slat = sample_tex_slat(
                shape_result, conditioning, device,
                seed=seed,
                tex_guidance_strength=tex_guidance_strength,
                tex_sampling_steps=tex_sampling_steps,
            )
            source = "full"
            log.info(
                f"SegviGen: encode complete — source=full, "
                f"shape_slat={shape_slat.feats.shape}, tex_slat={tex_slat.feats.shape}"
            )
        except Exception as e:
            log.warning(f"SegviGen: tex sampling failed ({e}); using shape_only mode")
            tex_slat = None
            subs = None
            source = "shape_only"

        mm.soft_empty_cache()
        return ({
            "latent": shape_slat,
            "tex_slat": tex_slat,
            "subs": subs,
            "voxel": {"resolution": resolution},
            "source": source,
        },)


class SegviGenFromShapeResult:
    """
    Convert a TRELLIS2_SHAPE_RESULT directly into a SEGVIGEN_SLAT.

    This is the CORRECT input path for SegviGen. Instead of creating a zero-
    feature SparseTensor from a binary voxel grid (SegviGenVoxelEncode), this
    node extracts the shape_slat that TRELLIS2 already computed during shape
    generation.

    Why this matters:
    - shape_slat carries real geometry features (not zeros).
    - shape_slat is at 512-resolution, matching the model's APE coordinate space.
    - concat_cond in the sampler uses these features → proper shape conditioning.

    Connect: TRELLIS2 "Image to Shape" → shape_result → this node → slat
    """

    CATEGORY = "SegviGen"
    FUNCTION = "encode"
    RETURN_TYPES = ("SEGVIGEN_SLAT",)
    RETURN_NAMES = ("slat",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shape_result": ("TRELLIS2_SHAPE_RESULT", {
                    "tooltip": "Connect the shape_result output from TRELLIS2 Image-to-Shape node.",
                }),
            },
        }

    def encode(self, shape_result: dict):
        import sys
        import os
        import torch
        import comfy.model_management as mm

        check_interrupt()

        # Resolve SparseTensor class from TRELLIS2
        for p in sys.path:
            candidate = os.path.join(p, "trellis2", "sparse", "__init__.py")
            if os.path.isfile(candidate):
                break
        else:
            trellis2_nodes = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "..", "ComfyUI-TRELLIS2", "nodes")
            )
            if trellis2_nodes not in sys.path:
                sys.path.insert(0, trellis2_nodes)
        from trellis2.modules.sparse import SparseTensor

        device = mm.get_torch_device()

        slat_data = shape_result.get("shape_slat")
        if slat_data is None:
            raise ValueError(
                "SegviGenFromShapeResult: shape_result has no 'shape_slat'. "
                "Connect to the shape_result output of the TRELLIS2 Image-to-Shape node."
            )

        # shape_result['shape_slat'] is either an IPC-serialised dict or a live SparseTensor
        if isinstance(slat_data, dict) and slat_data.get("_type") == "SparseTensor":
            feats = slat_data["feats"].to(device=device, dtype=torch.float32)
            coords = slat_data["coords"].to(device=device)
            shape_slat = SparseTensor(feats=feats, coords=coords)
        elif hasattr(slat_data, "feats") and hasattr(slat_data, "coords"):
            # Already a live SparseTensor
            shape_slat = SparseTensor(
                feats=slat_data.feats.to(device=device, dtype=torch.float32),
                coords=slat_data.coords.to(device=device),
            )
        else:
            raise ValueError(
                f"SegviGenFromShapeResult: unknown shape_slat format: {type(slat_data)}"
            )

        N = shape_slat.feats.shape[0]
        # TRELLIS2 shape_slat is always 512-res; fall back to shape_result metadata
        resolution = shape_result.get("resolution", 512)

        log.info(
            f"SegviGenFromShapeResult: {N} voxels, "
            f"resolution={resolution}, "
            f"feature_norm={shape_slat.feats.norm(dim=-1).mean():.4f}"
        )
        mm.soft_empty_cache()
        return ({"latent": shape_slat, "voxel": {"resolution": resolution},
                 "source": "shape_only", "tex_slat": None, "subs": None},)
