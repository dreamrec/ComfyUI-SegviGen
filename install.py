"""
SegviGen checkpoint installer — thin CLI wrapper around core/checkpoints.py.

All checkpoint logic lives in core/checkpoints.py. This file exists only
for backward compatibility (ComfyUI calls install.py on first load) and
as a CLI entry point for manual downloads.
"""
import os
import logging

log = logging.getLogger("segvigen")


def ensure_checkpoint(models_dir: str) -> str:
    """Legacy wrapper — delegates to core/checkpoints.py."""
    from core.checkpoints import resolve_checkpoint
    return resolve_checkpoint("full_legacy")


def ensure_interactive_checkpoint(models_dir: str) -> str:
    """Legacy wrapper — delegates to core/checkpoints.py."""
    from core.checkpoints import resolve_checkpoint
    return resolve_checkpoint("interactive_binary")


def ensure_encoder_checkpoint(models_dir: str) -> str:
    """Legacy alias for ensure_checkpoint."""
    return ensure_checkpoint(models_dir)


if __name__ == "__main__":
    import argparse
    import sys

    # Ensure package root is on sys.path for core imports
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Download SegviGen checkpoints")
    parser.add_argument("--mode", choices=["interactive", "full", "full-2d-guided", "all"],
                        default="all", help="Which checkpoint(s) to download")
    parser.add_argument("--models-dir", default=None,
                        help="Path to models/segvigen directory (auto-detected from ComfyUI)")
    args = parser.parse_args()

    from core.checkpoints import resolve_checkpoint, CHECKPOINT_MANIFEST

    modes = {
        "interactive": ["interactive_binary"],
        "full": ["full"],
        "full-2d-guided": ["full_2d_guided"],
        "all": ["interactive_binary", "full", "full_2d_guided"],
    }

    for mode in modes[args.mode]:
        try:
            path = resolve_checkpoint(mode)
            print(f"[{mode}] ready: {path}")
        except Exception as e:
            print(f"[{mode}] FAILED: {e}")
