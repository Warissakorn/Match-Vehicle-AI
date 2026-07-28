"""Tests for filename timestamp parsing (no filesystem / model needed)."""

from datetime import datetime

from mash_reid import timestamp_ocr
from mash_reid.frame_loader import load_frames, parse_timestamp, parse_timestamp_from_name


def test_parse_underscore_format():
    assert parse_timestamp_from_name("A_20260723_101530.jpg") == datetime(2026, 7, 23, 10, 15, 30)


def test_parse_with_camera_prefix():
    assert parse_timestamp_from_name("cam1-20260723_101530.png") == datetime(2026, 7, 23, 10, 15, 30)


def test_parse_dashed_format():
    assert parse_timestamp_from_name("2026-07-23_10-15-30.jpeg") == datetime(2026, 7, 23, 10, 15, 30)


def test_parse_iso_like_format():
    assert parse_timestamp_from_name("2026-07-23T10:15:30.png") == datetime(2026, 7, 23, 10, 15, 30)


def test_parse_14_digit_format():
    assert parse_timestamp_from_name("20260723101530.jpg") == datetime(2026, 7, 23, 10, 15, 30)


def test_no_timestamp_returns_none():
    assert parse_timestamp_from_name("random_photo.jpg") is None


def test_invalid_date_is_rejected():
    # Month 13 matches the shape but is not a real date -> None.
    assert parse_timestamp_from_name("A_20261332_101530.jpg") is None


def test_full_path_uses_basename_only():
    result = parse_timestamp_from_name("/data/frames/A_20260723_101530.jpg")
    assert result == datetime(2026, 7, 23, 10, 15, 30)


# --- parse_timestamp: OCR sidecar takes priority over every other source ----

def test_ocr_sidecar_wins_over_filename():
    # The filename claims 10:15:30, but the OCR'd on-screen clock says 09:00:00.
    ocr_map = {"A_20260723_101530.jpg": datetime(2026, 7, 23, 9, 0, 0)}
    ts, source = parse_timestamp("/data/frames/A_20260723_101530.jpg", ocr_map)
    assert (ts, source) == (datetime(2026, 7, 23, 9, 0, 0), "ocr")


def test_no_sidecar_entry_falls_back_to_filename():
    ts, source = parse_timestamp("/data/frames/A_20260723_101530.jpg", {})
    assert (ts, source) == (datetime(2026, 7, 23, 10, 15, 30), "filename")


def test_none_sidecar_falls_back_to_filename():
    ts, source = parse_timestamp("/data/frames/A_20260723_101530.jpg", None)
    assert (ts, source) == (datetime(2026, 7, 23, 10, 15, 30), "filename")


def test_load_frames_uses_ocr_sidecar(tmp_path):
    # load_frames only lists matching extensions and resolves each timestamp
    # -- it never decodes the image itself, so a real image isn't needed.
    name = "random_photo.jpg"  # no timestamp in the filename
    (tmp_path / name).write_bytes(b"not a real image, doesn't need to be")

    ts = datetime(2026, 7, 23, 9, 30, 0)
    timestamp_ocr._save_sidecar(str(tmp_path), {name: ts})

    frames = load_frames(str(tmp_path), "A")
    assert len(frames) == 1
    assert frames[0].timestamp == ts
    assert frames[0].timestamp_source == "ocr"


def test_load_frames_without_sidecar_falls_back_cleanly(tmp_path):
    (tmp_path / "A_20260723_101530.jpg").write_bytes(b"not a real image")

    frames = load_frames(str(tmp_path), "A")
    assert len(frames) == 1
    assert frames[0].timestamp_source == "filename"


# --- timestamp_source reflects which sidecar wrote the times -----------------

def test_timeline_sidecar_labels_frames_as_timeline(tmp_path):
    name = "A_20260723_101530_000123.jpg"
    (tmp_path / name).write_bytes(b"not a real image, never decoded")
    timestamp_ocr.write_sidecar_doc(str(tmp_path), {
        "version": 2, "source": "timeline",
        "frames": {name: "2026-07-23T09:30:00.500000"},
    })

    frames = load_frames(str(tmp_path), "A")
    assert len(frames) == 1
    assert frames[0].timestamp_source == "timeline"
    assert frames[0].timestamp == datetime(2026, 7, 23, 9, 30, 0, 500000)


def test_legacy_sidecar_still_labels_frames_as_ocr(tmp_path):
    name = "A_20260723_101530.jpg"
    (tmp_path / name).write_bytes(b"not a real image")
    timestamp_ocr._save_sidecar(str(tmp_path), {name: datetime(2026, 7, 23, 9, 0, 0)})

    frames = load_frames(str(tmp_path), "A")
    assert frames[0].timestamp_source == "ocr"


def test_parse_timestamp_source_label_is_overridable():
    ocr_map = {"A_20260723_101530.jpg": datetime(2026, 7, 23, 9, 0, 0)}
    _ts, source = parse_timestamp("/d/A_20260723_101530.jpg", ocr_map, "timeline")
    assert source == "timeline"
