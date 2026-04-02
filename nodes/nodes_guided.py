"""
SegviGen 2D-guided conditioning node.

Extracts conditioning from a 2D segmentation map (color labels) for
guided full-mesh segmentation. Uses the same TRELLIS conditioning
pipeline but with task_mode="full_2d_guided" so the full sampler
selects the correct checkpoint (full_seg_w_2d_map.ckpt).

Does NOT use BiRefNet by default — segmentation maps have hard color
boundaries that BiRefNet would damage. Uses a palette-safe crop path.
"""
import logging

from .helpers import check_interrupt

log = logging.getLogger("segvigen")


class SegviGenGet2DMapConditioning:
    """
    Extract 2D segmentation map conditioning for guided full segmentation.

    Accepts a pre-segmented 2D image (color map) where each color represents
    a desired part. The conditioning guides the 3D segmentation to match
    the 2D part layout.

    The segmentation map is:
    1. Cropped/padded to square (no content-aware masking)
    2. Resized to 512x512 using nearest-neighbor (preserves hard color edges)
    3. Passed through the same DINOv3 conditioning pipeline
    4. Returned with task_mode="full_2d_guided" so the sampler selects
       the correct checkpoint
    """

    CATEGORY = "SegviGen"
    FUNCTION = "condition"
    RETURN_TYPES = ("SEGVIGEN_COND",)
    RETURN_NAMES = ("conditioning",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_config": ("TRELLIS2_MODEL_CONFIG", {
                    "tooltip": "Config from Load TRELLIS2 Models node",
                }),
                "segmentation_map": ("IMAGE", {
                    "tooltip": "2D segmentation map — each color = one desired 3D part.",
                }),
            },
            "optional": {
                "mask": ("MASK",),
                "preserve_palette": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Preserve input color palette in the 3D segmentation.",
                }),
            },
        }

    def condition(self, model_config, segmentation_map, mask=None,
                  preserve_palette=True):
        import torch
        import comfy.model_management as mm

        check_interrupt()

        resolution = model_config.get("resolution", "1024_cascade")
        include_1024 = resolution in ("1024_cascade", "1536_cascade", "1024")

        # ── Palette-safe preprocessing ───────────────────────────────────
        # Do NOT use BiRefNet — segmentation maps have hard color boundaries.
        # Crop/pad to square, resize to 512 with nearest-neighbor.
        image = segmentation_map
        B, H, W, C = image.shape

        # Crop to square (center crop)
        if H != W:
            size = min(H, W)
            y0 = (H - size) // 2
            x0 = (W - size) // 2
            image = image[:, y0:y0+size, x0:x0+size, :]

        # Resize to 512x512 with nearest-neighbor to preserve palette
        if image.shape[1] != 512 or image.shape[2] != 512:
            image = torch.nn.functional.interpolate(
                image.permute(0, 3, 1, 2),  # [B,C,H,W]
                size=(512, 512),
                mode="nearest",
            ).permute(0, 2, 3, 1)  # back to [B,H,W,C]

        # Build mask if not provided (all foreground)
        if mask is None:
            mask = torch.ones(B, 512, 512, dtype=torch.float32)
        elif mask.shape[1] != 512 or mask.shape[2] != 512:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(1),
                size=(512, 512),
                mode="nearest",
            ).squeeze(1)

        # ── Extract palette from unique colors ───────────────────────────
        palette = None
        if preserve_palette:
            pixels = (image[0].cpu().numpy() * 255).astype("uint8")
            unique_colors = set()
            for row in pixels.reshape(-1, C):
                unique_colors.add(tuple(row[:3].tolist()))
            palette = [c for c in unique_colors if sum(c) > 30]
            log.info(f"SegviGen 2D-guided: extracted {len(palette)} palette colors")

        log.info(f"SegviGen: extracting 2D-guided conditioning "
                 f"(resolution={resolution}, palette_colors={len(palette) if palette else 0})")

        # ── Run TRELLIS conditioning ─────────────────────────────────────
        from nodes.nodes_conditioning import _load_trellis2_stages
        stages = _load_trellis2_stages()
        cond, _ = stages.run_conditioning(
            model_config=model_config,
            image=image,
            mask=mask,
            include_1024=include_1024,
            background_color="black",
        )

        from core.contracts import build_segvigen_cond, TASK_FULL_2D_GUIDED

        mm.soft_empty_cache()
        return (build_segvigen_cond(
            cond["cond_512"], cond["neg_cond"],
            cond_1024=cond.get("cond_1024"),
            task_mode=TASK_FULL_2D_GUIDED,
            preserve_palette=preserve_palette,
            palette=palette,
        ),)
