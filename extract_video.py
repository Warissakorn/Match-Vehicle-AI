"""Extract timestamped still frames from a video for point A or B.

Feed a raw camera video and get a folder of frames the Re-ID pipeline can use.
The video's start time is taken from its filename (e.g. A_20260723_101500.mp4);
each frame is timestamped as start + frame_index / fps.

Examples:
    python extract_video.py --video A_20260723_101500.mp4 --point A
    python extract_video.py --video cam_b.mp4 --point B \\
        --interval 0.5 --start-time "2026-07-23 10:17:45" --out frames/B
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

# Make ``config`` (project root) and the ``mash_reid`` package importable.
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from mash_reid import video_extractor  # noqa: E402

_START_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y%m%d_%H%M%S", "%Y-%m-%dT%H:%M:%S"]


def _parse_start_time(value: str) -> datetime:
    for fmt in _START_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Unrecognized --start-time '{value}'. Try 'YYYY-MM-DD HH:MM:SS'."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract timestamped frames from a video.")
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument("--point", required=True, choices=["A", "B"], help="Camera point label")
    parser.add_argument("--out", help="Output folder (default: <video>_frames_<point> next to video)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between extracted frames (default 1.0)")
    parser.add_argument("--start-time", type=_parse_start_time, default=None,
                        help="Override the video start time (else parsed from filename, then mtime)")
    parser.add_argument("--ext", default="jpg", help="Output image extension (default jpg)")
    args = parser.parse_args(argv)

    out_dir = args.out or video_extractor.default_output_dir(args.video, args.point)
    start, source = video_extractor.resolve_start_time(args.video, args.start_time)
    print(f"Video start time: {start:%Y-%m-%d %H:%M:%S}  (source: {source})")
    print(f"Extracting every {args.interval}s into: {out_dir}")

    def progress(done, total, msg):
        total_str = str(total) if total else "?"
        print(f"  [{done}/{total_str}] {msg}", end="\r", flush=True)

    written = video_extractor.extract_frames(
        args.video, out_dir, args.point,
        interval_seconds=args.interval, start_time=args.start_time,
        image_ext=args.ext, progress=progress,
    )
    print(f"\nDone. Wrote {len(written)} frames to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
