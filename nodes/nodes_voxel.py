"""
SegviGen voxel pipeline nodes:
  - SegviGenGLBtoVoxel: mesh → SEGVIGEN_VOXEL
  - SegviGenVoxelEncode: SEGVIGEN_VOXEL → SEGVIGEN_SLAT
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
        import struct
        import json
        import numpy as np
        import torch
        import comfy.model_management as mm
        import folder_paths
        import os
        import sys

        check_interrupt()

        # ── resolve SparseTensor from TRELLIS2 ──────────────────────────
        _trellis2_nodes = None
        for p in sys.path:
            candidate = os.path.join(p, "trellis2", "sparse", "__init__.py")
            if os.path.isfile(candidate):
                _trellis2_nodes = p
                break
        if _trellis2_nodes is None:
            # common path relative to this file
            _trellis2_nodes = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..", "..", "ComfyUI-TRELLIS2", "nodes")
            )
            if _trellis2_nodes not in sys.path:
                sys.path.insert(0, _trellis2_nodes)
        from trellis2.sparse import SparseTensor

        # ── read in_channels from safetensors header (no full load) ─────
        ckpt_path = os.path.join(folder_paths.models_dir, "segvigen", "full_seg.safetensors")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"SegviGen checkpoint not found: {ckpt_path}")
        with open(ckpt_path, "rb") as _f:
            _header_size = struct.unpack("<Q", _f.read(8))[0]
            _header = json.loads(_f.read(_header_size))
        # out_layer maps model_channels → out_channels (the denoising latent dim).
        # in_channels = out_channels + concat_cond_channels; noise is only out_channels.
        in_channels = _header["flow_model.out_layer.weight"]["shape"][0]

        # ── build SparseTensor from voxel occupancy ──────────────────────
        grid = voxel["grid"]               # bool np.ndarray [R, R, R]
        coords_np = np.argwhere(grid).astype(np.int32)  # [N, 3]
        if len(coords_np) == 0:
            raise ValueError("SegviGenVoxelEncode: voxel grid is empty")

        device = mm.get_torch_device()
        dtype_str = model_config.get("dtype", "fp16")
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]

        # coords: [N, 4] (batch_idx=0, x, y, z)
        batch_col = np.zeros((len(coords_np), 1), dtype=np.int32)
        coords = torch.from_numpy(np.concatenate([batch_col, coords_np], axis=1)).to(device)

        # Zero features — sampler's _add_noise turns this into pure Gaussian noise
        feats = torch.zeros(len(coords), in_channels, dtype=dtype, device=device)
        latent = SparseTensor(feats=feats, coords=coords)

        # Propagate resolution so downstream nodes don't default to 64
        voxel_with_res = dict(voxel)
        voxel_with_res["resolution"] = int(grid.shape[0])

        log.info(f"SegviGen: encoded {len(coords)} voxels → SLAT ({in_channels}ch, R={grid.shape[0]})")
        mm.soft_empty_cache()
        return ({"latent": latent, "voxel": voxel_with_res},)
