"""
SegviGen checkpoint installer.

Downloads the SegviGen segmentation flow model checkpoint from HuggingFace
to ComfyUI/models/segvigen/ on first use.

Checkpoint: Aero-Ex/SegviGen on HuggingFace (same source as the Aero-Ex port).
The specific filename must be verified against the actual HuggingFace repo.
"""
import os
import logging

log = logging.getLogger("segvigen")

HF_REPO = "Aero-Ex/SegviGen"
HF_REPO_INTERACTIVE = "fenghora/SegviGen"

# Full segmentation checkpoint (automatic mode)
CHECKPOINT_FILENAME = "full_seg.safetensors"
# Interactive segmentation checkpoint (point-guided mode)
INTERACTIVE_CHECKPOINT = "interactive_seg.ckpt"


def ensure_checkpoint(models_dir: str) -> str:
    """
    Return path to the SegviGen checkpoint, downloading if not present.

    Args:
        models_dir: path to ComfyUI/models/segvigen/

    Returns:
        Absolute path to the checkpoint .safetensors file.

    Raises:
        RuntimeError: if download fails.
    """
    ckpt_path = os.path.join(models_dir, CHECKPOINT_FILENAME)

    if os.path.exists(ckpt_path):
        return ckpt_path

    log.info(f"SegviGen: checkpoint not found at {ckpt_path}, downloading from HuggingFace...")
    os.makedirs(models_dir, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
        # Capture the returned path — hf_hub_download may place files in a
        # cache subdirectory depending on the huggingface_hub version; always
        # use the returned path rather than our pre-computed ckpt_path.
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=CHECKPOINT_FILENAME,
            local_dir=models_dir,
            local_dir_use_symlinks=False,
        )
        log.info(f"SegviGen: checkpoint downloaded to {downloaded_path}")
    except Exception as e:
        raise RuntimeError(
            f"SegviGen: failed to download checkpoint.\n"
            f"Manual download: https://huggingface.co/{HF_REPO}\n"
            f"Place '{CHECKPOINT_FILENAME}' in: {models_dir}\n"
            f"Error: {e}"
        ) from e

    return downloaded_path


def ensure_interactive_checkpoint(models_dir: str) -> str:
    """
    Return path to the interactive SegviGen checkpoint, downloading if not present.

    This is the point-guided binary segmentation model (interactive_seg.ckpt, ~7.86 GB)
    from fenghora/SegviGen on HuggingFace.
    """
    ckpt_path = os.path.join(models_dir, INTERACTIVE_CHECKPOINT)

    if os.path.exists(ckpt_path):
        return ckpt_path

    log.info(f"SegviGen: interactive checkpoint not found at {ckpt_path}, downloading...")
    os.makedirs(models_dir, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_INTERACTIVE,
            filename=INTERACTIVE_CHECKPOINT,
            local_dir=models_dir,
            local_dir_use_symlinks=False,
        )
        log.info(f"SegviGen: interactive checkpoint downloaded to {downloaded_path}")
    except Exception as e:
        raise RuntimeError(
            f"SegviGen: failed to download interactive checkpoint.\n"
            f"Manual download: https://huggingface.co/{HF_REPO_INTERACTIVE}\n"
            f"Place '{INTERACTIVE_CHECKPOINT}' in: {models_dir}\n"
            f"Error: {e}"
        ) from e

    return downloaded_path


def ensure_encoder_checkpoint(models_dir: str) -> str:
    """
    Return path to the SegviGen checkpoint that contains the shape encoder.

    The encoder weights are NOT in a separate file — they are embedded in the
    main full_seg.safetensors checkpoint under encoder key prefixes.
    This function is an alias for ensure_checkpoint().
    """
    return ensure_checkpoint(models_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download SegviGen checkpoint")
    parser.add_argument("--models-dir", default=None,
                        help="Path to models/segvigen directory (auto-detected from ComfyUI)")
    args = parser.parse_args()

    if args.models_dir:
        models_dir = args.models_dir
    else:
        try:
            import folder_paths
            models_dir = os.path.join(folder_paths.models_dir, "segvigen")
        except ImportError:
            models_dir = os.path.join(os.path.dirname(__file__), "models")

    path = ensure_checkpoint(models_dir)
    print(f"Checkpoint ready: {path}")
