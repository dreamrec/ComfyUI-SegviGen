"""
Checkpoint source definitions and download logic for SegviGen.

This module owns ALL checkpoint manifests and download helpers.
It replaces and extends the basic download functions in install.py.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Dict, Optional

log = logging.getLogger("segvigen")

# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

CHECKPOINT_MANIFEST: Dict[str, dict] = {
    "interactive_binary": {
        "hf_repo": "fenghora/SegviGen",
        "filename": "interactive_seg.ckpt",
        "format": "ckpt",
        "sha256": "11e41fdaa0c03f581c0c44eb8f9b27604c0d5e4d2794cb02b6edff80c23f4dd1",
        "description": "Point-guided interactive binary segmentation",
    },
    "full": {
        "hf_repo": "fenghora/SegviGen",
        "filename": "full_seg.ckpt",
        "format": "ckpt",
        "sha256": "c71618e45b1671026da66cf9875cc5c2d41d5a894b01a15ac5b019946e6a207c",
        "description": "Automatic full-mesh part segmentation",
    },
    "full_2d_guided": {
        "hf_repo": "fenghora/SegviGen",
        "filename": "full_seg_w_2d_map.ckpt",
        "format": "ckpt",
        "sha256": None,  # not yet downloaded from fenghora
        "description": "2D-map-guided full-mesh segmentation",
    },
    "full_legacy": {
        "hf_repo": "Aero-Ex/SegviGen",
        "filename": "full_seg.safetensors",
        "format": "safetensors",
        "sha256": "61508a7dfc163a95d977ac30c1d3f9ac8f258b663ed1fcea67b3a748b3beaa65",
        "description": "Legacy Aero-Ex full segmentation (safetensors)",
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_models_dir() -> str:
    """Return ``ComfyUI/models/segvigen/``, creating the directory if needed.

    Uses ``folder_paths.models_dir`` (lazy-imported so this module can be
    imported outside of a running ComfyUI process for testing).
    """
    import folder_paths  # noqa: E402  — lazy; requires ComfyUI on sys.path

    models_dir = os.path.join(folder_paths.models_dir, "segvigen")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def verify_checkpoint_hash(path: str, expected_sha256: Optional[str]) -> bool:
    """Compute SHA-256 of *path* and compare to *expected_sha256*.

    Returns ``True`` if the hashes match **or** if *expected_sha256* is
    ``None`` (hash not yet recorded in the manifest).
    """
    if expected_sha256 is None:
        return True

    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            sha.update(chunk)

    actual = sha.hexdigest()
    if actual != expected_sha256:
        log.warning(
            "SegviGen: hash mismatch for %s — expected %s, got %s",
            path,
            expected_sha256,
            actual,
        )
        return False
    return True


def download_checkpoint(task_mode: str, models_dir: Optional[str] = None) -> str:
    """Download the checkpoint for *task_mode* and return its absolute path.

    Parameters
    ----------
    task_mode:
        Key into :data:`CHECKPOINT_MANIFEST`.
    models_dir:
        Target directory.  Defaults to :func:`get_models_dir` when ``None``.

    Returns
    -------
    str
        Absolute path to the downloaded checkpoint file.

    Raises
    ------
    ValueError
        If *task_mode* is not in the manifest.
    RuntimeError
        If the download or hash verification fails.
    """
    if task_mode not in CHECKPOINT_MANIFEST:
        raise ValueError(
            f"Unknown task_mode '{task_mode}'. "
            f"Available: {list(CHECKPOINT_MANIFEST.keys())}"
        )

    entry = CHECKPOINT_MANIFEST[task_mode]
    hf_repo = entry["hf_repo"]
    filename = entry["filename"]
    expected_sha = entry["sha256"]

    if models_dir is None:
        models_dir = get_models_dir()
    os.makedirs(models_dir, exist_ok=True)

    log.info(
        "SegviGen: downloading '%s' from %s/%s …",
        task_mode,
        hf_repo,
        filename,
    )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download checkpoints. "
            "Install it with:  pip install huggingface_hub"
        ) from exc

    try:
        downloaded_path = hf_hub_download(
            repo_id=hf_repo,
            filename=filename,
            local_dir=models_dir,
            local_dir_use_symlinks=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"SegviGen: failed to download checkpoint for '{task_mode}'.\n"
            f"Manual download: https://huggingface.co/{hf_repo}\n"
            f"Place '{filename}' in: {models_dir}\n"
            f"Error: {exc}"
        ) from exc

    if not verify_checkpoint_hash(downloaded_path, expected_sha):
        raise RuntimeError(
            f"SegviGen: SHA-256 verification failed for '{task_mode}' "
            f"({downloaded_path}).  The file may be corrupt — please "
            f"delete it and retry."
        )

    log.info("SegviGen: checkpoint ready at %s", downloaded_path)
    return downloaded_path


def resolve_checkpoint(task_mode: str) -> str:
    """Return the absolute path to the checkpoint for *task_mode*.

    If the file is not already present on disk it will be downloaded
    automatically via :func:`download_checkpoint`.

    Raises
    ------
    ValueError
        If *task_mode* is not in the manifest.
    RuntimeError
        If the download fails.
    """
    if task_mode not in CHECKPOINT_MANIFEST:
        raise ValueError(
            f"Unknown task_mode '{task_mode}'. "
            f"Available: {list(CHECKPOINT_MANIFEST.keys())}"
        )

    entry = CHECKPOINT_MANIFEST[task_mode]
    models_dir = get_models_dir()
    ckpt_path = os.path.join(models_dir, entry["filename"])

    if os.path.isfile(ckpt_path):
        log.debug("SegviGen: checkpoint for '%s' found at %s", task_mode, ckpt_path)
        return ckpt_path

    return download_checkpoint(task_mode, models_dir)


def list_available_checkpoints() -> Dict[str, dict]:
    """Return a dict mapping each *task_mode* to its status.

    Each value is a dict with keys:

    * ``path`` — absolute path (``str``) if the file exists, else ``None``
    * ``exists`` — ``bool``
    * ``manifest`` — the raw manifest entry
    """
    models_dir = get_models_dir()
    result: Dict[str, dict] = {}

    for task_mode, entry in CHECKPOINT_MANIFEST.items():
        ckpt_path = os.path.join(models_dir, entry["filename"])
        exists = os.path.isfile(ckpt_path)
        result[task_mode] = {
            "path": ckpt_path if exists else None,
            "exists": exists,
            "manifest": entry,
        }

    return result
