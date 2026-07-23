"""Extract timestamped still frames from a video for one camera point.

Lets you feed raw videos of points A and B instead of pre-cut still frames.
Each extracted frame is written with a filename the rest of the system already
understands (``A_20260723_101530_000123.jpg``), so ``frame_loader`` picks up
its capture time and the temporal gate keeps working.

Timestamp model (per the chosen design): the **video's start time** comes from
its filename (e.g. ``A_20260723_101500.mp4``); each frame's real-world time is
then ``start_time + frame_index / fps``. If the filename carries no timestamp,
we fall back to the file's modification time, and an explicit override is always
allowed.

Frames are sampled every ``interval_seconds`` (default 1 s). ``cv2`` is a
deferred import so the pure-logic helpers below unit-test without OpenCV.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import config
from mash_reid.frame_loader import parse_timestamp_from_name


def resolve_start_time(video_path: str, explicit: datetime | None = None) -> tuple[datetime, str]:
    """Resolve the real-world start time of a video, returning (time, source).

    Priority: explicit override > timestamp parsed from filename > file mtime.
    """
    if explicit is not None:
        return explicit, "explicit"
    ts = parse_timestamp_from_name(video_path)
    if ts is not None:
        return ts, "filename"
    mtime = os.path.getmtime(video_path)
    return datetime.fromtimestamp(mtime, tz=timezone.utc).replace(tzinfo=None), "mtime"


def frame_step(fps: float, interval_seconds: float) -> int:
    """How many source frames to advance between two saved frames (>= 1)."""
    if fps <= 0 or interval_seconds <= 0:
        return 1
    return max(1, round(fps * interval_seconds))


def frame_timestamp(start_time: datetime, frame_index: int, fps: float) -> datetime:
    """Real-world time of frame ``frame_index`` given the video start and fps."""
    offset = frame_index / fps if fps > 0 else 0.0
    return start_time + timedelta(seconds=offset)


def frame_filename(point: str, ts: datetime, frame_index: int, ext: str = "jpg") -> str:
    """Filename encoding the timestamp (second precision) plus a unique index.

    The zero-padded ``frame_index`` keeps names unique when several frames fall
    in the same second; ``frame_loader`` still parses the ``YYYYMMDD_HHMMSS``
    part for the capture time.
    """
    ext = ext.lstrip(".")
    return f"{point}_{ts:%Y%m%d_%H%M%S}_{frame_index:06d}.{ext}"


def extract_frames(
    video_path: str,
    output_dir: str,
    point: str,
    interval_seconds: float = 1.0,
    start_time: datetime | None = None,
    image_ext: str = "jpg",
    progress=None,
) -> list[str]:
    """Extract frames from ``video_path`` into ``output_dir`` for ``point``.

    Returns the list of written frame paths. ``progress`` is an optional
    callable ``(done, total, message)`` for GUIs/CLIs.
    """
    # Validate inputs before importing the heavy dep, so bad paths fail cleanly
    # even where OpenCV isn't installed.
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    import cv2  # deferred so module import stays light / OpenCV-free

    os.makedirs(output_dir, exist_ok=True)

    start, _src = resolve_start_time(video_path, start_time)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            fps = 30.0  # sensible default when the container omits fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = frame_step(fps, interval_seconds)

        written: list[str] = []
        frame_index = 0
        saved = 0
        est_total = (total_frames // step) if total_frames > 0 else 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                ts = frame_timestamp(start, frame_index, fps)
                name = frame_filename(point, ts, frame_index, image_ext)
                out_path = os.path.join(output_dir, name)
                cv2.imwrite(out_path, frame)
                written.append(out_path)
                saved += 1
                if progress:
                    progress(saved, est_total, name)
            frame_index += 1

        return written
    finally:
        cap.release()


def default_output_dir(video_path: str, point: str) -> str:
    """A reasonable default frames folder next to the video."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    parent = os.path.dirname(os.path.abspath(video_path))
    return os.path.join(parent, f"{base}_frames_{point}")
