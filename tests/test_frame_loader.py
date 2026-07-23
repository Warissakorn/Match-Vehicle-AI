"""Tests for filename timestamp parsing (no filesystem / model needed)."""

from datetime import datetime

from mash_reid.frame_loader import parse_timestamp_from_name


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
