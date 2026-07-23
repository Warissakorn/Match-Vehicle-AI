"""Mash-Object-AI: cross-point vehicle Re-Identification.

Match the same physical vehicle between two camera points (A and B) using
visual appearance only. See the top-level README for usage.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "frame_loader",
    "video_extractor",
    "detector",
    "embedder",
    "matcher",
    "pipeline",
]
