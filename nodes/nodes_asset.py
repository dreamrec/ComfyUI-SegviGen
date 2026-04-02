"""
SegviGen asset-native nodes — the paper's original encoding pipeline.

  - SegviGenAssetPrepare: mesh -> normalize -> voxelize -> .vxz
  - SegviGenAssetEncode: .vxz -> shape_enc -> tex_enc -> shape_dec -> SLAT
"""
import logging
import os

from .helpers import check_interrupt

log = logging.getLogger("segvigen")


class SegviGenAssetPrepare:
    """
    Prepare a mesh asset for SegviGen's native encoding pipeline.

    Normalizes the mesh to [-0.5, 0.5], preprocesses textures to square
    power-of-two, runs o_voxel flexible dual-grid conversion, and writes
    a .vxz file for encoding.
    """

    CATEGORY = "SegviGen"
    FUNCTION = "prepare"
    RETURN_TYPES = ("SEGVIGEN_VOXEL",)
    RETURN_NAMES = ("voxel",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
            "optional": {
                "texture_resolution": ("INT", {
                    "default": 1024, "min": 256, "max": 4096, "step": 256,
                    "tooltip": "Texture map resolution for preprocessing.",
                }),
                "voxel_resolution": ("INT", {
                    "default": 512, "min": 64, "max": 1024,
                    "tooltip": "Voxel grid resolution for dual-grid conversion.",
                }),
            },
        }

    def prepare(self, trimesh, texture_resolution=1024, voxel_resolution=512):
        import folder_paths
        from core.asset_encode import prepare_asset_to_vxz
        from core.contracts import build_segvigen_voxel

        check_interrupt()

        vxz_dir = os.path.join(folder_paths.output_directory, "segvigen", "vxz")
        os.makedirs(vxz_dir, exist_ok=True)
        vxz_path = os.path.join(vxz_dir, f"asset_{id(trimesh)}.vxz")

        log.info(f"SegviGen asset prepare: voxelizing at {voxel_resolution}^3")
        vxz_path, normalization = prepare_asset_to_vxz(
            trimesh, vxz_path,
            texture_resolution=texture_resolution,
            voxel_resolution=voxel_resolution,
        )

        return (build_segvigen_voxel(
            resolution=voxel_resolution,
            vxz_path=vxz_path,
            normalization=normalization,
        ),)


class SegviGenAssetEncode:
    """
    Encode a prepared asset into SEGVIGEN_SLAT using native TRELLIS2 encoders.

    Reads the .vxz file, runs shape_encoder + tex_encoder + shape_decoder
    to produce real shape_slat, tex_slat, and subs — full paper fidelity.
    """

    CATEGORY = "SegviGen"
    FUNCTION = "encode"
    RETURN_TYPES = ("SEGVIGEN_SLAT",)
    RETURN_NAMES = ("slat",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voxel": ("SEGVIGEN_VOXEL",),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def encode(self, voxel, seed=0):
        import torch
        import comfy.model_management as mm
        from core.asset_encode import vxz_to_latent_slat
        from core.contracts import build_segvigen_slat, SOURCE_ASSET_FULL

        check_interrupt()

        vxz_path = voxel.get("vxz_path")
        if vxz_path is None or not os.path.isfile(vxz_path):
            raise ValueError(
                f"SegviGenAssetEncode: no .vxz file found at {vxz_path}. "
                "Connect SegviGenAssetPrepare to produce a valid .vxz file."
            )

        device = str(mm.get_torch_device())
        torch.manual_seed(seed)

        log.info(f"SegviGen asset encode: encoding {vxz_path}")
        shape_slat, meshes, subs, tex_slat = vxz_to_latent_slat(vxz_path, device)

        mm.soft_empty_cache()
        return (build_segvigen_slat(
            shape_slat,
            tex_slat=tex_slat,
            subs=subs,
            voxel_resolution=voxel.get("resolution", 512),
            source=SOURCE_ASSET_FULL,
            normalization=voxel.get("normalization"),
        ),)
