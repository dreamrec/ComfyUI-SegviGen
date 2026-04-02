"""
SegviGen asset-native encoding backend.

Ports the paper's original process_glb_to_vxz and vxz_to_latent_slat
pipeline for real mesh-to-latent encoding without the TRELLIS2 bridge.
"""
import logging
import os
import sys
import torch
import numpy as np
from PIL import Image

log = logging.getLogger("segvigen")

# Module-level model cache (loaded once, reused across runs)
_model_cache = {
    "shape_encoder": None,
    "tex_encoder": None,
    "shape_decoder": None,
}


def _ensure_o_voxel_path():
    """Add TRELLIS2's o_voxel to sys.path if not already importable."""
    try:
        import o_voxel
        return
    except ImportError:
        pass
    # Search for o_voxel inside TRELLIS2's environment
    import folder_paths
    candidates = [
        os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-TRELLIS2", "_env_trellis2", "Lib", "site-packages"),
        os.path.join(folder_paths.base_path, "custom_nodes", "ComfyUI-TRELLIS2", "nodes"),
    ]
    for p in candidates:
        if os.path.isdir(os.path.join(p, "o_voxel")):
            if p not in sys.path:
                sys.path.insert(0, p)
                log.info(f"SegviGen: added o_voxel path: {p}")
            return
    raise ImportError(
        "SegviGen: o_voxel library not found. "
        "Ensure ComfyUI-TRELLIS2 is installed with its CUDA extensions."
    )


def make_texture_square_pow2(img: Image.Image, target_size=None):
    """Resize texture to square power-of-two dimensions (max 2048)."""
    w, h = img.size
    max_side = max(w, h)
    pow2 = 1
    while pow2 < max_side:
        pow2 *= 2
    if target_size is not None:
        pow2 = target_size
    pow2 = min(pow2, 2048)
    return img.resize((pow2, pow2), Image.BILINEAR)


def preprocess_scene_textures(asset):
    """Resize all scene textures to square power-of-two for o_voxel compatibility."""
    import trimesh
    if not isinstance(asset, trimesh.Scene):
        return asset
    TEX_KEYS = ["baseColorTexture", "normalTexture",
                "metallicRoughnessTexture", "emissiveTexture",
                "occlusionTexture"]
    for geom in asset.geometry.values():
        visual = getattr(geom, "visual", None)
        mat = getattr(visual, "material", None)
        if mat is None:
            continue
        for key in TEX_KEYS:
            if not hasattr(mat, key):
                continue
            tex = getattr(mat, key)
            if tex is None:
                continue
            if isinstance(tex, Image.Image):
                setattr(mat, key, make_texture_square_pow2(tex))
            elif hasattr(tex, "image") and tex.image is not None:
                img = tex.image
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                tex.image = make_texture_square_pow2(img)
        if hasattr(mat, "image") and mat.image is not None:
            img = mat.image
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            mat.image = make_texture_square_pow2(img)
    return asset


def normalize_scene(asset):
    """Normalize mesh/scene to [-0.5, 0.5] cube. Returns (asset, center, scale, aabb)."""
    aabb = asset.bounding_box.bounds
    center = (aabb[0] + aabb[1]) / 2
    scale = 0.99999 / (aabb[1] - aabb[0]).max()
    asset.apply_translation(-center)
    asset.apply_scale(scale)
    return asset, center.tolist(), float(scale), aabb.tolist()


def prepare_asset_to_vxz(
    trimesh_or_path,
    output_path: str,
    texture_resolution: int = 1024,
    voxel_resolution: int = 512,
):
    """
    Full paper-native asset preparation: mesh -> .vxz file.

    Returns:
        (vxz_path, normalization_dict)
    """
    import trimesh as _tm
    _ensure_o_voxel_path()
    import o_voxel

    # Load mesh
    if isinstance(trimesh_or_path, str):
        asset = _tm.load(trimesh_or_path, force='scene')
    elif isinstance(trimesh_or_path, _tm.Scene):
        asset = trimesh_or_path
    elif isinstance(trimesh_or_path, _tm.Trimesh):
        asset = _tm.Scene(geometry={'mesh': trimesh_or_path})
    else:
        asset = _tm.Scene(geometry={'mesh': trimesh_or_path})

    # Preprocess textures
    asset = preprocess_scene_textures(asset)

    # Normalize to [-0.5, 0.5]
    asset, center, scale, aabb = normalize_scene(asset)

    # Convert to single mesh for voxelization
    mesh = asset.to_mesh() if isinstance(asset, _tm.Scene) else asset
    vertices = torch.from_numpy(np.asarray(mesh.vertices)).float()
    faces = torch.from_numpy(np.asarray(mesh.faces)).long()

    log.info(f"SegviGen asset: voxelizing {len(vertices)} vertices, {len(faces)} faces "
             f"at resolution {voxel_resolution}")

    # Flexible dual-grid conversion
    voxel_indices, dual_vertices, intersected = (
        o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices, faces, grid_size=voxel_resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            face_weight=1.0, boundary_weight=0.2,
            regularization_weight=1e-2, timing=False
        )
    )

    # Sort by space-filling curve
    vid = o_voxel.serialize.encode_seq(voxel_indices)
    mapping = torch.argsort(vid)
    voxel_indices = voxel_indices[mapping]
    dual_vertices = dual_vertices[mapping]
    intersected = intersected[mapping]

    # Textured mesh to volumetric attributes
    voxel_indices_mat, attributes = (
        o_voxel.convert.textured_mesh_to_volumetric_attr(
            asset, grid_size=voxel_resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            timing=False
        )
    )
    vid_mat = o_voxel.serialize.encode_seq(voxel_indices_mat)
    mapping_mat = torch.argsort(vid_mat)
    attributes = {k: v[mapping_mat] for k, v in attributes.items()}

    # Pack dual_vertices and intersected into attributes
    dual_vertices = dual_vertices * voxel_resolution - voxel_indices
    dual_vertices = (torch.clamp(dual_vertices, 0, 1) * 255).type(torch.uint8)
    intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] +
                   4 * intersected[:, 2:3]).type(torch.uint8)

    attributes['dual_vertices'] = dual_vertices
    attributes['intersected'] = intersected

    # Write .vxz file
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    o_voxel.io.write(output_path, voxel_indices, attributes)

    n_voxels = len(voxel_indices)
    log.info(f"SegviGen asset: wrote {n_voxels} voxels to {output_path}")

    normalization = {
        "center": center,
        "scale": scale,
        "aabb": aabb,
        "resolution": voxel_resolution,
    }
    return output_path, normalization


def _load_native_models(device="cuda"):
    """Load and cache TRELLIS2 shape/tex encoder and shape decoder."""
    if _model_cache["shape_encoder"] is not None:
        return _model_cache["shape_encoder"], _model_cache["tex_encoder"], _model_cache["shape_decoder"]

    try:
        from trellis2 import models
    except ImportError:
        # Try loading from TRELLIS2 nodes path
        from core.trellis2_shim import load_trellis2_stages
        stages = load_trellis2_stages()
        stages._init_config()
        # The models module should now be importable
        from trellis2 import models

    log.info("SegviGen asset: loading native encoder/decoder models...")

    shape_enc = models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/shape_enc_next_dc_f16c32_fp16"
    ).to(device).eval()

    tex_enc = models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/tex_enc_next_dc_f16c32_fp16"
    ).to(device).eval()

    shape_dec = models.from_pretrained(
        "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16"
    ).to(device).eval()

    _model_cache["shape_encoder"] = shape_enc
    _model_cache["tex_encoder"] = tex_enc
    _model_cache["shape_decoder"] = shape_dec

    log.info("SegviGen asset: native models loaded and cached")
    return shape_enc, tex_enc, shape_dec


def vxz_to_latent_slat(vxz_path: str, device: str = "cuda"):
    """
    Encode .vxz file to latent SLAT using native TRELLIS2 encoders.

    Returns:
        (shape_slat, meshes, subs, tex_slat)
    """
    _ensure_o_voxel_path()
    import o_voxel
    from trellis2.modules import sparse as sp

    shape_enc, tex_enc, shape_dec = _load_native_models(device)

    # Read .vxz
    coords, data = o_voxel.io.read(vxz_path)
    coords = torch.cat([
        torch.zeros(coords.shape[0], 1, dtype=torch.int32),
        coords
    ], dim=1).to(device)

    # Decode dual_vertices and intersected
    vertices = (data['dual_vertices'].to(device).float() / 255)
    intersected_raw = data['intersected'].to(device)
    intersected = torch.cat([
        intersected_raw % 2,
        intersected_raw // 2 % 2,
        intersected_raw // 4 % 2,
    ], dim=-1).bool().float()

    vertices_sparse = sp.SparseTensor(vertices, coords)
    intersected_sparse = sp.SparseTensor(intersected, coords)

    # Shape encoding + decoding
    with torch.no_grad():
        shape_slat = shape_enc(vertices_sparse, intersected_sparse)
        shape_slat = sp.SparseTensor(shape_slat.feats.to(device), shape_slat.coords.to(device))
        shape_dec.set_resolution(512)
        meshes, subs = shape_dec(shape_slat, return_subs=True)

    # Texture encoding
    base_color = (data['base_color'].float() / 255)
    metallic = (data['metallic'].float() / 255)
    roughness = (data['roughness'].float() / 255)
    alpha = (data['alpha'].float() / 255)
    attr = torch.cat([base_color, metallic, roughness, alpha], dim=-1).to(device) * 2 - 1

    with torch.no_grad():
        tex_slat = tex_enc(sp.SparseTensor(attr, coords))

    n = shape_slat.feats.shape[0]
    log.info(f"SegviGen asset: encoded {n} voxels — shape_slat {shape_slat.feats.shape}, "
             f"tex_slat {tex_slat.feats.shape}")

    return shape_slat, meshes, subs, tex_slat


def map_points_via_tex_encoder(points_xyz_512: list, coords, device: str = "cuda"):
    """
    Map click points through the tex_encoder for paper-faithful
    interactive point-coordinate transformation.

    This is the upstream approach from inference_interactive.py:
    create zero-feature sparse points at click coords, run through
    tex_encoder, use the returned coords as latent-space point positions.

    Args:
        points_xyz_512: list of [x, y, z] in 512-space
        coords: existing SLAT coords (unused, kept for API consistency)
        device: torch device

    Returns:
        torch.Tensor [N, 4] int32 — mapped coordinates (batch_idx, x, y, z)
    """
    from trellis2.modules import sparse as sp

    _, tex_enc, _ = _load_native_models(device)

    vxz_coords = torch.tensor(points_xyz_512, dtype=torch.int32).to(device)
    # Add batch dimension
    vxz_coords = torch.cat([
        torch.zeros((vxz_coords.shape[0], 1), dtype=torch.int32, device=device),
        vxz_coords
    ], dim=1)

    # Run through tex_encoder with zero features (6 channels for PBR attrs)
    zero_feats = torch.zeros((vxz_coords.shape[0], 6), dtype=torch.float32, device=device)
    with torch.no_grad():
        encoded = tex_enc(sp.SparseTensor(zero_feats, vxz_coords))

    # Use the returned coords as the mapped point positions
    mapped_coords = torch.unique(encoded.coords, dim=0)

    log.info(f"SegviGen asset: mapped {len(points_xyz_512)} points -> "
             f"{len(mapped_coords)} latent coords via tex_encoder")

    return mapped_coords
