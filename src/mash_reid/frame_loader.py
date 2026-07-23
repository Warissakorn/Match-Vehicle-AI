"""Load still frames from a folder and attach a timestamp to each.

The timestamp is what powers the temporal gate in matching (a vehicle passes A
before it passes B). We derive it, in priority order, from:

    1. The filename, using the regex/format pairs in ``config.TIMESTAMP_PATTERNS``.
    2. EXIF ``DateTimeOriginal`` (if the image carries it).
    3. The file modification time (always available fallback).

Only ``parse_timestamp`` touches the filesystem for EXIF; the pure filename
logic is isolated so it can be unit-tested without any real files.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import config

# Pre-compile the filename patterns once.
_COMPILED_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(rx), fmt) for rx, fmt in config.TIMESTAMP_PATTERNS
]


@dataclass
class Frame:
    """A single still image plus its resolved capture time and source point."""

    path: str
    timestamp: datetime
    point: str  # "A" or "B"
    timestamp_source: str  # "filename" | "exif" | "mtime"

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def parse_timestamp_from_name(filename: str) -> datetime | None:
    """Return the timestamp encoded in ``filename`` or ``None`` if none matches.

    Pure function: takes only the string, no filesystem access. The first
    pattern in ``config.TIMESTAMP_PATTERNS`` that matches wins.
    """
    base = os.path.basename(filename)
    for pattern, fmt in _COMPILED_PATTERNS:
        m = pattern.search(base)
        if not m:
            continue
        try:
            return datetime.strptime(m.group(1), fmt)
        except ValueError:
            # Matched the shape but not a real date (e.g. month 13) -> keep trying.
            continue
    return None


def _parse_timestamp_from_exif(path: str) -> datetime | None:
    """Read EXIF DateTimeOriginal (tag 36867) if present; else ``None``."""
    try:
        from PIL import Image  # local import so tests don't require Pillow
    except ImportError:
        return None
    try:
        with Image.open(path) as img:
            exif = img.getexif()
        if not exif:
            return None
        # 36867 = DateTimeOriginal, 306 = DateTime
        for tag in (36867, 306):
            raw = exif.get(tag)
            if raw:
                return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None
    return None


def parse_timestamp(path: str) -> tuple[datetime, str]:
    """Resolve a timestamp for ``path``, returning (timestamp, source)."""
    ts = parse_timestamp_from_name(path)
    if ts is not None:
        return ts, "filename"

    ts = _parse_timestamp_from_exif(path)
    if ts is not None:
        return ts, "exif"

    mtime = os.path.getmtime(path)
    return datetime.fromtimestamp(mtime, tz=timezone.utc).replace(tzinfo=None), "mtime"


def load_frames(folder: str, point: str) -> list[Frame]:
    """Load all supported images from ``folder`` as ``Frame`` objects.

    Frames are returned sorted by timestamp (ascending), which is the natural
    order for downstream temporal reasoning.
    """
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Not a directory: {folder}")

    frames: list[Frame] = []
    for entry in sorted(os.listdir(folder)):
        ext = os.path.splitext(entry)[1].lower()
        if ext not in config.IMAGE_EXTENSIONS:
            continue
        path = os.path.join(folder, entry)
        if not os.path.isfile(path):
            continue
        ts, source = parse_timestamp(path)
        frames.append(Frame(path=path, timestamp=ts, point=point, timestamp_source=source))

    frames.sort(key=lambda f: f.timestamp)
    return frames
