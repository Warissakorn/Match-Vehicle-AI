"""Tests for the video-frame extractor helpers.

These cover the pure-logic pieces (start-time resolution, frame stepping,
per-frame timestamp, filenames) without needing OpenCV or a real video —
``cv2`` is only imported inside ``extract_frames``.
"""

import os
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


# --- Tests that need OpenCV (skipped where cv2 isn't installed, e.g. CI) ----

def _make_video(cv2, np, path, n_frames=30, fps=10, size=(64, 48)):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), (i * 8) % 255, np.uint8))
    writer.release()


def test_write_image_creates_file(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    frame = np.full((48, 64, 3), 120, np.uint8)
    out = str(tmp_path / "frame.jpg")
    ve._write_image(cv2, out, frame, "jpg")
    assert os.path.getsize(out) > 0


def test_write_image_handles_non_ascii_path(tmp_path):
    # The bug this fixes: cv2.imwrite silently fails on non-ASCII (e.g. Thai)
    # paths on Windows. The imencode+open path must produce a real file.
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    thai_dir = tmp_path / "กล้องจุดA"
    thai_dir.mkdir()
    frame = np.full((48, 64, 3), 200, np.uint8)
    out = str(thai_dir / "A_20260723_101500_000000.jpg")
    ve._write_image(cv2, out, frame, "jpg")
    assert os.path.getsize(out) > 0


def test_extract_frames_writes_files_into_non_ascii_output(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    vid = str(tmp_path / "A_20260723_101500.mp4")
    _make_video(cv2, np, vid)
    out_dir = str(tmp_path / "ผลลัพธ์จุดA")  # non-ASCII output folder
    written = ve.extract_frames(vid, out_dir, "A", interval_seconds=1.0)
    assert len(written) == 3
    assert all(os.path.getsize(p) > 0 for p in written)
