"""Tests for OCR timestamp parsing and the sidecar cache.

``parse_overlay_text`` is pure (no OCR, no image I/O) and gets the bulk of
the coverage here, since that's where the actual accuracy of "read the clock
burned into the frame" lives. The imaging side (``ocr_folder`` etc.) is
exercised with a fake ``reader`` object so these tests don't need the
``easyocr`` package installed.
"""

from datetime import datetime

import numpy as np
import pytest

from mash_reid import timestamp_ocr
from mash_reid.timestamp_ocr import parse_overlay_text


# --- parse_overlay_text: clean text -----------------------------------------

def test_iso_date_colon_time():
    assert parse_overlay_text("2026-07-23 10:15:30") == datetime(2026, 7, 23, 10, 15, 30)


def test_iso_date_slash_separator():
    assert parse_overlay_text("2026/07/23 10:15:30") == datetime(2026, 7, 23, 10, 15, 30)


def test_iso_date_dot_separator():
    assert parse_overlay_text("2026.07.23 10:15:30") == datetime(2026, 7, 23, 10, 15, 30)


def test_day_first_slash_date():
    assert parse_overlay_text("23/07/2026 10:15:30") == datetime(2026, 7, 23, 10, 15, 30)


def test_day_first_dash_date():
    assert parse_overlay_text("23-07-2026 10:15:30") == datetime(2026, 7, 23, 10, 15, 30)


def test_month_first_date_disambiguated_by_validity():
    # day=07/month=23 is impossible -> falls back to month=07/day=23.
    assert parse_overlay_text("07-23-2026 10:15:30") == datetime(2026, 7, 23, 10, 15, 30)


def test_compact_date():
    assert parse_overlay_text("20260723 10:15:30") == datetime(2026, 7, 23, 10, 15, 30)


def test_am_pm_morning():
    assert parse_overlay_text("2026-07-23 10:15:30 AM") == datetime(2026, 7, 23, 10, 15, 30)


def test_am_pm_afternoon():
    assert parse_overlay_text("2026-07-23 01:15:30 PM") == datetime(2026, 7, 23, 13, 15, 30)


def test_12am_is_midnight():
    assert parse_overlay_text("2026-07-23 12:00:00 AM") == datetime(2026, 7, 23, 0, 0, 0)


def test_12pm_is_noon():
    assert parse_overlay_text("2026-07-23 12:00:00 PM") == datetime(2026, 7, 23, 12, 0, 0)


def test_extra_surrounding_text_ignored():
    assert parse_overlay_text("CAM01 2026-07-23 10:15:30 REC") == datetime(2026, 7, 23, 10, 15, 30)


def test_dash_separated_time():
    assert parse_overlay_text("2026-07-23 10-15-30") == datetime(2026, 7, 23, 10, 15, 30)


# --- parse_overlay_text: OCR noise ------------------------------------------

def test_ocr_confuses_o_for_zero():
    assert parse_overlay_text("2O26-O7-23 1O:15:3O") == datetime(2026, 7, 23, 10, 15, 30)


def test_ocr_confuses_l_for_one():
    assert parse_overlay_text("2026-07-23 l0:l5:30") == datetime(2026, 7, 23, 10, 15, 30)


def test_ocr_confuses_i_for_one():
    assert parse_overlay_text("2026-07-23 I0:I5:30") == datetime(2026, 7, 23, 10, 15, 30)


# --- parse_overlay_text: rejects garbage -------------------------------------

def test_empty_string_is_none():
    assert parse_overlay_text("") is None


def test_non_date_text_is_none():
    assert parse_overlay_text("hello world") is None


def test_invalid_month_is_none():
    assert parse_overlay_text("2026-13-23 10:15:30") is None


def test_invalid_hour_is_none():
    assert parse_overlay_text("2026-07-23 25:15:30") is None


def test_date_without_time_is_none():
    assert parse_overlay_text("2026-07-23") is None


# --- sidecar read/write -------------------------------------------------------

def test_load_sidecar_missing_file_returns_empty(tmp_path):
    assert timestamp_ocr.load_sidecar(str(tmp_path)) == {}


def test_load_sidecar_corrupt_file_returns_empty(tmp_path):
    (tmp_path / timestamp_ocr.config.TIMESTAMP_SIDECAR_NAME).write_text("not json")
    assert timestamp_ocr.load_sidecar(str(tmp_path)) == {}


def test_sidecar_round_trip(tmp_path):
    ts = {"A_0001.jpg": datetime(2026, 7, 23, 10, 15, 30)}
    timestamp_ocr._save_sidecar(str(tmp_path), ts)
    assert timestamp_ocr.load_sidecar(str(tmp_path)) == ts


# --- imaging side, with a fake reader (no easyocr installed) -----------------

_BBOX = [[5, 5], [100, 5], [100, 20], [5, 20]]


class _FakeReader:
    """Mimics easyocr.Reader.readtext for a fixed sequence of calls.

    ``script`` is a list of per-call results, each a list of (bbox, text,
    conf) triples -- the same shape ``easyocr`` returns with ``detail=1``.
    ``detail=0`` calls (used by ``read_timestamp``) get just the text field.
    """

    def __init__(self, script: list[list[tuple]]):
        self._script = script
        self.calls = 0

    def readtext(self, image, detail=1):
        idx = min(self.calls, len(self._script) - 1) if self._script else 0
        result = self._script[idx] if self._script else []
        self.calls += 1
        if detail == 0:
            return [text for _bbox, text, _conf in result]
        return result


def _blank_image(h=40, w=200):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_find_overlay_region_returns_bbox_of_matching_text():
    reader = _FakeReader([[(_BBOX, "2026-07-23 10:15:30", 0.9)]])
    region = timestamp_ocr.find_overlay_region(_blank_image(), reader)
    assert region is not None
    x1, y1, x2, y2 = region
    assert x1 < 5 and y1 < 5 and x2 > 100 and y2 > 20


def test_find_overlay_region_none_when_nothing_parses():
    reader = _FakeReader([[([[0, 0], [10, 0], [10, 10], [0, 10]], "junk", 0.5)]])
    assert timestamp_ocr.find_overlay_region(_blank_image(), reader) is None


def test_read_timestamp_joins_detected_lines():
    reader = _FakeReader([[(_BBOX, "2026-07-23", 0.9), (_BBOX, "10:15:30", 0.9)]])
    ts = timestamp_ocr.read_timestamp(_blank_image(), reader)
    assert ts == datetime(2026, 7, 23, 10, 15, 30)


def test_read_timestamp_none_when_unparseable():
    reader = _FakeReader([[(_BBOX, "not a date", 0.5)]])
    assert timestamp_ocr.read_timestamp(_blank_image(), reader) is None


def test_ocr_folder_writes_sidecar_and_returns_mapping(tmp_path):
    import cv2

    names = [f"A_{i:04d}.jpg" for i in range(3)]
    for name in names:
        cv2.imwrite(str(tmp_path / name), _blank_image())

    # Call sequence: [0] region-find on frame 0, [1..3] per-frame reads,
    # one second apart.
    script = [
        [(_BBOX, "2026-07-23 10:15:30", 0.9)],  # region-find
        [(_BBOX, "2026-07-23 10:15:30", 0.9)],  # frame 0
        [(_BBOX, "2026-07-23 10:15:31", 0.9)],  # frame 1
        [(_BBOX, "2026-07-23 10:15:32", 0.9)],  # frame 2
    ]
    reader = _FakeReader(script)

    results = timestamp_ocr.ocr_folder(str(tmp_path), names, reader=reader)

    assert results == {
        "A_0000.jpg": datetime(2026, 7, 23, 10, 15, 30),
        "A_0001.jpg": datetime(2026, 7, 23, 10, 15, 31),
        "A_0002.jpg": datetime(2026, 7, 23, 10, 15, 32),
    }
    assert timestamp_ocr.load_sidecar(str(tmp_path)) == results


def test_ocr_folder_skips_unreadable_images(tmp_path):
    (tmp_path / "broken.jpg").write_bytes(b"")  # zero-byte -> unreadable
    reader = _FakeReader([])
    results = timestamp_ocr.ocr_folder(str(tmp_path), ["broken.jpg"], reader=reader)
    assert results == {}


def test_ocr_folder_no_results_writes_no_sidecar(tmp_path):
    import cv2

    name = "A_0000.jpg"
    cv2.imwrite(str(tmp_path / name), _blank_image())
    reader = _FakeReader([[(_BBOX, "junk", 0.5)]])
    results = timestamp_ocr.ocr_folder(str(tmp_path), [name], reader=reader)
    assert results == {}
    assert not (tmp_path / timestamp_ocr.config.TIMESTAMP_SIDECAR_NAME).exists()
