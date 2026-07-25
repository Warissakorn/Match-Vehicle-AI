"""Extract timestamped still frames from one or more videos for point A or B.

Feed raw camera video(s) and get a folder of frames the Re-ID pipeline can use.
Each video's start time is taken from its own filename (e.g.
A_20260723_101500.mp4); each frame is timestamped as start + frame_index / fps.

Examples:
    python extract_video.py --video A_20260723_101500.mp4 --point A
    python extract_video.py --video cam_b.mp4 --point B \\
        --interval 0.5 --start-time "2026-07-23 10:17:45" --out frames/B

    # Multiple clips for the same point (e.g. the camera splits every hour),
    # combined into one output folder:
    python extract_video.py --video A_part1.mp4 A_part2.mp4 A_part3.mp4 --point A
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

# Make ``config`` (project root) and the ``mash_reid`` package importable.
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from mash_reid import logging_setup, video_extractor  # noqa: E402

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
    parser.add_argument("--video", required=True, nargs="+",
                        help="Path to one or more input video files. Multiple "
                             "videos are treated as clips of the same point and "
                             "combined into one output folder.")
    parser.add_argument("--point", required=True, choices=["A", "B"], help="Camera point label")
    parser.add_argument("--out", help="Output folder (default: <video>_frames_<point> next to "
                                       "video, or <point>_frames next to the first clip)")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Seconds between extracted frames (default 1.0)")
    parser.add_argument("--start-time", type=_parse_start_time, default=None,
                        help="Override the video start time (else parsed from filename, then "
                             "mtime). Only valid with a single --video -- each clip in a "
                             "multi-clip run resolves its own start time.")
    parser.add_argument("--ext", default="jpg", help="Output image extension (default jpg)")
    parser.add_argument("--ocr-time", action="store_true",
                        help="Read each extracted frame's true timestamp via OCR "
                             "afterwards (writes a sidecar the pipeline picks up)")
    parser.add_argument("--log-dir", default=logging_setup.DEFAULT_LOG_DIR,
                        help="Folder for run log files (default: logs/)")
    parser.add_argument("--verbose", action="store_true",
                        help="Also print DEBUG detail to the console")
    args = parser.parse_args(argv)

    log_path = logging_setup.setup_logging(
        args.log_dir, console_level=logging.DEBUG if args.verbose else logging.INFO)
    print(f"Logging to {log_path}")

    def progress(done, total, msg):
        total_str = str(total) if total else "?"
        print(f"  [{done}/{total_str}] {msg}", end="\r", flush=True)

    if len(args.video) == 1:
        video = args.video[0]
        out_dir = args.out or video_extractor.default_output_dir(video, args.point)
        start, source = video_extractor.resolve_start_time(video, args.start_time)
        print(f"Video start time: {start:%Y-%m-%d %H:%M:%S}  (source: {source})")
        print(f"Extracting every {args.interval}s into: {out_dir}")
        written = video_extractor.extract_frames(
            video, out_dir, args.point,
            interval_seconds=args.interval, start_time=args.start_time,
            image_ext=args.ext, progress=progress,
        )
    else:
        if args.start_time is not None:
            parser.error("--start-time is only valid with a single --video; "
                         "each clip in a multi-clip run resolves its own start time.")
        out_dir = args.out or video_extractor.default_output_dir_multi(args.video, args.point)
        print(f"Extracting {len(args.video)} clip(s) every {args.interval}s into: {out_dir}")
        written = video_extractor.extract_many(
            args.video, out_dir, args.point,
            interval_seconds=args.interval, image_ext=args.ext, progress=progress,
        )
    print(f"\nDone. Wrote {len(written)} frames to {out_dir}")

    if args.ocr_time and written:
        from mash_reid import timestamp_ocr

        names = sorted(os.path.basename(p) for p in written)
        print(f"OCR-ing timestamps for {len(names)} frame(s) ...")
        results = timestamp_ocr.ocr_folder(out_dir, names, progress=progress)
        print(f"\n  Read {len(results)}/{len(names)} timestamp(s) from the frame overlay.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
