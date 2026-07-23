"""Tests for the video-frame extractor helpers.

These cover the pure-logic pieces (start-time resolution, frame stepping,
per-frame timestamp, filenames) without needing OpenCV or a real video —
``cv2`` is only imported inside ``extract_frames``.
"""

from datetime import datetime

import pytest

from mash_reid import frame_loader, video_extractor as ve


def test_resolve_start_time_from_filename():
    ts, source = ve.resolve_start_time("A_20260723_101500.mp4")
    assert ts == datetime(2026, 7, 23, 10, 15, 0)
    assert source == "filename"


def test_resolve_start_time_explicit_wins():
    explicit = datetime(2020, 1, 1, 0, 0, 0)
    ts, source = ve.resolve_start_time("A_20260723_101500.mp4", explicit)
    assert ts == explicit
    assert source == "explicit"


def test_frame_step_rounds_fps_times_interval():
    assert ve.frame_step(30.0, 1.0) == 30
    assert ve.frame_step(30.0, 0.5) == 15
    assert ve.frame_step(24.0, 2.0) == 48


def test_frame_step_never_below_one():
    assert ve.frame_step(30.0, 0.0) == 1
    assert ve.frame_step(0.0, 1.0) == 1
    assert ve.frame_step(1.0, 0.01) == 1


def test_frame_timestamp_advances_by_seconds():
    start = datetime(2026, 7, 23, 10, 0, 0)
    # 60 frames at 30 fps = 2 seconds in.
    assert ve.frame_timestamp(start, 60, 30.0) == datetime(2026, 7, 23, 10, 0, 2)


def test_frame_filename_encodes_timestamp_and_index():
    ts = datetime(2026, 7, 23, 10, 15, 30)
    name = ve.frame_filename("A", ts, 123, "jpg")
    assert name == "A_20260723_101530_000123.jpg"


def test_frame_filename_roundtrips_through_frame_loader():
    # A generated filename must be parseable back to its second-precision time.
    ts = datetime(2026, 7, 23, 10, 15, 30)
    name = ve.frame_filename("B", ts, 5, "png")
    assert frame_loader.parse_timestamp_from_name(name) == ts


def test_frame_filename_strips_leading_dot_in_ext():
    ts = datetime(2026, 7, 23, 10, 15, 30)
    assert ve.frame_filename("A", ts, 0, ".jpeg").endswith(".jpeg")


def test_default_output_dir_uses_video_basename():
    out = ve.default_output_dir("/data/A_20260723_101500.mp4", "A")
    assert out.endswith("A_20260723_101500_frames_A")


def test_extract_frames_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ve.extract_frames("does_not_exist.mp4", "/tmp/out", "A")
