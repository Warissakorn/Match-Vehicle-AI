"""Tests for OCR timestamp parsing and the sidecar cache.

``parse_overlay_text`` is pure (no OCR, no image I/O) and gets the bulk of
the coverage here, since that's where the actual accuracy of "read the clock
burned into the frame" lives. The imaging side (``ocr_folder`` etc.) is
exercised with a fake ``reader`` object so these tests don't need the
``easyocr`` package installed.
"""

import os
from datetime import datetime

import numpy as np
import pytest

import config
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


def test_dot_separated_time():
    # A real overlay format that reached us via a user report: dashes for the
    # date, dots for the time ("2026-07-24 17.40.50"). Previously only colon
    # and dash time separators were recognized, so this parsed as None and a
    # whole scan came back with zero readable frames.
    assert parse_overlay_text("2026-07-24 17.40.50") == datetime(2026, 7, 24, 17, 40, 50)


def test_dot_separated_date_and_time():
    assert parse_overlay_text("2026.07.24 17.40.50") == datetime(2026, 7, 24, 17, 40, 50)


def test_dot_separated_time_with_am_pm():
    assert parse_overlay_text("2026-07-24 05.40.50 PM") == datetime(2026, 7, 24, 17, 40, 50)


def test_dot_separated_time_tolerates_ocr_noise():
    assert parse_overlay_text("2O26-O7-24 17.4O.5O") == datetime(2026, 7, 24, 17, 40, 50)


def test_dot_separated_time_with_surrounding_text():
    assert parse_overlay_text("CAM01 2026-07-24 17.40.50 REC") == datetime(2026, 7, 24, 17, 40, 50)


def test_dot_separated_date_does_not_get_mistaken_for_time():
    # Regression guard: the dot-time regex must not eat part of a dot-dated
    # date and leave the actual time unfound.
    assert parse_overlay_text("2026.07.24 10:15:30") == datetime(2026, 7, 24, 10, 15, 30)


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


# --- reader device resolution -------------------------------------------------
# These are the four cases the old `gpu = bool(device) and device != "cpu"`
# got wrong; cuda_available is injected so they run with neither torch nor
# easyocr installed.

def test_auto_uses_gpu_when_cuda_available():
    assert timestamp_ocr._reader_gpu_flag("auto", cuda_available=True) == ("cuda", True)


def test_auto_falls_back_to_cpu_without_cuda():
    # The old code said gpu=True here, because "auto" is a truthy string.
    assert timestamp_ocr._reader_gpu_flag("auto", cuda_available=False) == ("cpu", False)


def test_none_uses_gpu_when_cuda_available():
    # The old code said gpu=False here, because None is falsy -- so the CLI's
    # default never used the GPU even on a machine that had one.
    assert timestamp_ocr._reader_gpu_flag(None, cuda_available=True) == ("cuda", True)


def test_explicit_cpu_never_uses_gpu():
    assert timestamp_ocr._reader_gpu_flag("cpu", cuda_available=True) == ("cpu", False)


def test_explicit_cuda_index_counts_as_gpu():
    assert timestamp_ocr._reader_gpu_flag("cuda:1", cuda_available=False) == ("cuda:1", True)


# --- custom overlay pattern ---------------------------------------------------
# The escape hatch for overlay formats the tolerant built-in parser can't
# guess. Users write it as a token template ("YYYY-MO-DD HH.MI.SS"), not a
# raw regex, so most of the coverage here is about that template compiling
# to the right thing and rejecting the ways it can be malformed.

def test_compile_and_match_basic_pattern():
    pattern = timestamp_ocr.compile_custom_pattern("YYYY-MO-DD HH.MI.SS")
    ts = timestamp_ocr.parse_overlay_text("2026-07-24 17.40.50", pattern)
    assert ts == datetime(2026, 7, 24, 17, 40, 50)


def test_compile_pattern_with_slashes_and_12_hour_clock():
    pattern = timestamp_ocr.compile_custom_pattern("DD/MO/YYYY hh:MI:SS AP")
    ts = timestamp_ocr.parse_overlay_text("24/07/2026 05:40:50 PM", pattern)
    assert ts == datetime(2026, 7, 24, 17, 40, 50)


def test_compile_pattern_with_two_digit_year():
    pattern = timestamp_ocr.compile_custom_pattern("YY-MO-DD HH:MI:SS")
    ts = timestamp_ocr.parse_overlay_text("26-07-24 17:40:50", pattern)
    assert ts == datetime(2026, 7, 24, 17, 40, 50)


def test_custom_pattern_ignores_surrounding_junk():
    pattern = timestamp_ocr.compile_custom_pattern("YYYY-MO-DD HH.MI.SS")
    ts = timestamp_ocr.parse_overlay_text("CAM01 2026-07-24 17.40.50 FIXED LENS", pattern)
    assert ts == datetime(2026, 7, 24, 17, 40, 50)


def test_custom_pattern_tolerates_ocr_digit_noise():
    pattern = timestamp_ocr.compile_custom_pattern("YYYY-MO-DD HH.MI.SS")
    ts = timestamp_ocr.parse_overlay_text("2O26-O7-24 17.4O.5O", pattern)
    assert ts == datetime(2026, 7, 24, 17, 40, 50)


def test_custom_pattern_rejects_invalid_date():
    pattern = timestamp_ocr.compile_custom_pattern("YYYY-MO-DD HH:MI:SS")
    assert timestamp_ocr.parse_overlay_text("2026-13-24 17:40:50", pattern) is None


def test_custom_pattern_none_when_text_does_not_match():
    pattern = timestamp_ocr.compile_custom_pattern("YYYY-MO-DD HH:MI:SS")
    assert timestamp_ocr.parse_overlay_text("hello world", pattern) is None


def test_custom_pattern_falls_back_to_builtin_parser():
    # A pattern configured for the user's usual format shouldn't break parsing
    # of a frame whose OCR text happens to match a completely different, but
    # still valid, built-in layout.
    pattern = timestamp_ocr.compile_custom_pattern("YYYY-MO-DD HH.MI.SS")
    ts = timestamp_ocr.parse_overlay_text("2026-07-24 17:40:50", pattern)
    assert ts == datetime(2026, 7, 24, 17, 40, 50)


def test_compile_rejects_empty_pattern():
    with pytest.raises(ValueError):
        timestamp_ocr.compile_custom_pattern("")


def test_compile_rejects_missing_year():
    with pytest.raises(ValueError, match="year"):
        timestamp_ocr.compile_custom_pattern("MO-DD HH:MI:SS")


def test_compile_rejects_missing_month_or_day():
    with pytest.raises(ValueError, match="MO"):
        timestamp_ocr.compile_custom_pattern("YYYY-DD HH:MI:SS")


def test_compile_rejects_missing_hour():
    with pytest.raises(ValueError, match="hour"):
        timestamp_ocr.compile_custom_pattern("YYYY-MO-DD MI:SS")


def test_compile_rejects_missing_minute_or_second():
    with pytest.raises(ValueError, match="MI"):
        timestamp_ocr.compile_custom_pattern("YYYY-MO-DD HH:SS")


def test_compile_rejects_12_hour_without_ampm():
    with pytest.raises(ValueError, match="AP"):
        timestamp_ocr.compile_custom_pattern("YYYY-MO-DD hh:MI:SS")


def test_compile_rejects_duplicate_token():
    with pytest.raises(ValueError):
        timestamp_ocr.compile_custom_pattern("YYYY-MO-DD HH:HH:SS")


def test_compile_rejects_whitespace_only_pattern():
    with pytest.raises(ValueError):
        timestamp_ocr.compile_custom_pattern("   ")


# --- reparse_candidates: apply a new pattern without re-running OCR ---------

def test_reparse_uses_stored_text_no_ocr_needed():
    stale = [
        timestamp_ocr.TimestampCandidate(text="2026-07-24 17.40.50", timestamp=None,
                                         confidence=0.9, origin="joined"),
    ]
    pattern = timestamp_ocr.compile_custom_pattern("YYYY-MO-DD HH.MI.SS")
    fresh = timestamp_ocr.reparse_candidates(stale, pattern)
    assert fresh[0].timestamp == datetime(2026, 7, 24, 17, 40, 50)
    assert fresh[0].text == stale[0].text  # unchanged, just reparsed


def test_reparse_re_ranks_newly_parseable_candidate_first():
    candidates = [
        timestamp_ocr.TimestampCandidate(text="Fixed Lens", timestamp=None,
                                         confidence=0.99, origin="line"),
        timestamp_ocr.TimestampCandidate(text="2026-07-24 17.40.50", timestamp=None,
                                         confidence=0.40, origin="line"),
    ]
    pattern = timestamp_ocr.compile_custom_pattern("YYYY-MO-DD HH.MI.SS")
    reparsed = timestamp_ocr.reparse_candidates(candidates, pattern)
    assert reparsed[0].timestamp == datetime(2026, 7, 24, 17, 40, 50)


def test_reparse_without_pattern_uses_builtin():
    candidates = [timestamp_ocr.TimestampCandidate(
        text="2026-07-24 17:40:50", timestamp=None, confidence=0.9, origin="joined")]
    reparsed = timestamp_ocr.reparse_candidates(candidates)
    assert reparsed[0].timestamp == datetime(2026, 7, 24, 17, 40, 50)


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
    cv2 = pytest.importorskip("cv2")

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
    # ocr_folder's deferred `import cv2` happens unconditionally on entry
    # (not just when a real image is decoded), so this needs the guard too.
    pytest.importorskip("cv2")
    (tmp_path / "broken.jpg").write_bytes(b"")  # zero-byte -> unreadable
    reader = _FakeReader([])
    results = timestamp_ocr.ocr_folder(str(tmp_path), ["broken.jpg"], reader=reader)
    assert results == {}


def test_ocr_folder_no_results_writes_no_sidecar(tmp_path):
    cv2 = pytest.importorskip("cv2")

    name = "A_0000.jpg"
    cv2.imwrite(str(tmp_path / name), _blank_image())
    reader = _FakeReader([[(_BBOX, "junk", 0.5)]])
    results = timestamp_ocr.ocr_folder(str(tmp_path), [name], reader=reader)
    assert results == {}
    assert not (tmp_path / timestamp_ocr.config.TIMESTAMP_SIDECAR_NAME).exists()


# --- candidates: confidence is kept, not discarded ---------------------------

def test_candidates_rank_parseable_above_unparseable():
    reader = _FakeReader([[(_BBOX, "garbage", 0.99), (_BBOX, "2026-07-23 10:15:30", 0.40)]])
    cands = timestamp_ocr.read_timestamp_candidates(_blank_image(), reader)
    assert cands[0].timestamp == datetime(2026, 7, 23, 10, 15, 30)
    assert any(c.timestamp is None for c in cands)


def test_candidates_keep_unparseable_readings():
    # The user needs to see what the OCR actually produced to judge whether a
    # frame is legible at all -- an empty list tells them nothing.
    reader = _FakeReader([[(_BBOX, "2O26-O7-Z3 lO:l5", 0.5)]])
    cands = timestamp_ocr.read_timestamp_candidates(_blank_image(), reader)
    assert cands and any("2O26" in c.text for c in cands)


def test_candidates_rank_by_confidence_among_parseable():
    reader = _FakeReader([[(_BBOX, "2026-07-23 10:15:30", 0.30),
                           (_BBOX, "2026-07-23 11:15:30", 0.95)]])
    cands = timestamp_ocr.read_timestamp_candidates(_blank_image(), reader)
    parseable = [c for c in cands if c.timestamp is not None]
    assert parseable[0].confidence >= parseable[-1].confidence


def test_joined_candidate_confidence_is_the_weakest_line():
    # One badly-read glyph anywhere in the joined string can flip a digit, so
    # the minimum is the honest score -- not the mean.
    reader = _FakeReader([[(_BBOX, "2026-07-23", 0.9), (_BBOX, "10:15:30", 0.2)]])
    cands = timestamp_ocr.read_timestamp_candidates(_blank_image(), reader)
    joined = [c for c in cands if c.origin == "joined"]
    assert joined and joined[0].confidence == pytest.approx(0.2)


def test_candidates_empty_when_nothing_detected():
    reader = _FakeReader([[]])
    assert timestamp_ocr.read_timestamp_candidates(_blank_image(), reader) == []


def test_find_overlay_region_prefers_the_most_confident_match():
    far = [[200, 200], [280, 200], [280, 215], [200, 215]]
    reader = _FakeReader([[(_BBOX, "2026-07-23 10:15:30", 0.30),
                           (far, "2026-07-23 11:15:30", 0.95)]])
    region = timestamp_ocr.find_overlay_region(_blank_image(h=400, w=400), reader)
    assert region[0] >= 190  # came from the high-confidence bbox, not the first


# --- sidecar v2 ---------------------------------------------------------------

def _v2_doc(frames):
    return {"version": 2, "source": "timeline",
            "frames": {k: v.isoformat() for k, v in frames.items()}}


def test_v2_sidecar_round_trips(tmp_path):
    frames = {"a.jpg": datetime(2026, 7, 23, 10, 15, 30, 500000)}
    timestamp_ocr.write_sidecar_doc(str(tmp_path), _v2_doc(frames))
    assert timestamp_ocr.load_sidecar(str(tmp_path)) == frames


def test_v1_flat_sidecar_still_loads(tmp_path):
    # Sidecars already on a user's disk must keep working with no migration.
    ts = {"a.jpg": datetime(2026, 7, 23, 10, 15, 30)}
    timestamp_ocr._save_sidecar(str(tmp_path), ts)
    assert timestamp_ocr.load_sidecar(str(tmp_path)) == ts


def test_one_bad_entry_does_not_discard_the_rest(tmp_path):
    # The old loader wrapped the whole map in one try/except, so a single
    # malformed value threw away every timestamp in the folder.
    doc = _v2_doc({"good.jpg": datetime(2026, 7, 23, 10, 15, 30)})
    doc["frames"]["bad.jpg"] = "not-a-timestamp"
    timestamp_ocr.write_sidecar_doc(str(tmp_path), doc)
    loaded = timestamp_ocr.load_sidecar(str(tmp_path))
    assert set(loaded) == {"good.jpg"}


def test_source_is_timeline_for_v2(tmp_path):
    timestamp_ocr.write_sidecar_doc(str(tmp_path), _v2_doc({}))
    assert timestamp_ocr.load_sidecar_source(str(tmp_path)) == "timeline"


def test_source_is_ocr_for_v1(tmp_path):
    timestamp_ocr._save_sidecar(str(tmp_path), {"a.jpg": datetime(2026, 7, 23, 10, 0, 0)})
    assert timestamp_ocr.load_sidecar_source(str(tmp_path)) == "ocr"


def test_source_is_ocr_when_no_sidecar(tmp_path):
    assert timestamp_ocr.load_sidecar_source(str(tmp_path)) == "ocr"


# --- describe_sidecar (main-window step-2 status line) -------------------------
#
# Returns (level, template, values) rather than a finished sentence so the GUI
# can translate the template before substituting -- see app/i18n.py.

def _rendered(folder):
    """(level, the sentence a caller would show)."""
    level, template, values = timestamp_ocr.describe_sidecar(folder)
    return level, template.format(**values)


def test_describe_sidecar_reports_nothing_reviewed(tmp_path):
    level, text = _rendered(str(tmp_path))
    assert level == "none"
    assert "filename" in text


def test_describe_sidecar_confirmed_fit_is_ok(tmp_path):
    doc = _v2_doc({"a.jpg": datetime(2026, 7, 23, 10, 15, 30)})
    doc.update({"confirmed_by_user": True, "implied_fps": 29.97, "warnings": []})
    timestamp_ocr.write_sidecar_doc(str(tmp_path), doc)
    level, text = _rendered(str(tmp_path))
    assert level == "ok"
    assert "1 frame time(s)" in text
    assert "29.97 fps" in text


def test_describe_sidecar_unconfirmed_fit_warns(tmp_path):
    # Written by the CLI rather than confirmed in the review dialog: the times
    # are usable but nobody has actually looked at them, which is exactly the
    # distinction this line exists to make visible.
    doc = _v2_doc({"a.jpg": datetime(2026, 7, 23, 10, 15, 30)})
    doc.update({"confirmed_by_user": False, "warnings": []})
    timestamp_ocr.write_sidecar_doc(str(tmp_path), doc)
    level, text = _rendered(str(tmp_path))
    assert level == "warn"
    assert "not confirmed" in text


def test_describe_sidecar_warns_when_the_fit_flagged_problems(tmp_path):
    doc = _v2_doc({"a.jpg": datetime(2026, 7, 23, 10, 15, 30)})
    doc.update({"confirmed_by_user": True, "warnings": ["only 40% of readings agree"]})
    timestamp_ocr.write_sidecar_doc(str(tmp_path), doc)
    assert timestamp_ocr.describe_sidecar(str(tmp_path))[0] == "warn"


def test_describe_sidecar_handles_v1_flat_files(tmp_path):
    timestamp_ocr._save_sidecar(str(tmp_path), {"a.jpg": datetime(2026, 7, 23, 10, 0, 0),
                                                "b.jpg": datetime(2026, 7, 23, 10, 0, 1)})
    level, text = _rendered(str(tmp_path))
    assert level == "ok"
    assert "2 frame time(s)" in text
    assert "every-frame" in text


def test_describe_sidecar_ignores_a_non_numeric_fps(tmp_path):
    # A hand-edited or foreign file must not crash the main window's status
    # line -- it is repainted on every keystroke in the folder entry.
    doc = _v2_doc({"a.jpg": datetime(2026, 7, 23, 10, 15, 30)})
    doc.update({"confirmed_by_user": True, "implied_fps": "fast", "warnings": []})
    timestamp_ocr.write_sidecar_doc(str(tmp_path), doc)
    level, template, values = timestamp_ocr.describe_sidecar(str(tmp_path))
    assert level == "ok"
    assert "fps" not in template and "fps" not in values


def test_describe_sidecar_survives_a_corrupt_file(tmp_path):
    (tmp_path / config.TIMESTAMP_SIDECAR_NAME).write_text("{not json", encoding="utf-8")
    assert timestamp_ocr.describe_sidecar(str(tmp_path))[0] == "none"


def test_describe_sidecar_template_and_values_always_agree(tmp_path):
    """Every branch must return a template whose fields the values fill --
    a mismatch would raise mid-render in the one place that repaints on
    every keystroke."""
    import re as _re
    cases = [
        {},
        {"confirmed_by_user": True, "implied_fps": 25.0, "warnings": []},
        {"confirmed_by_user": True, "warnings": []},
        {"confirmed_by_user": False, "implied_fps": 30.0, "warnings": ["x"]},
        {"confirmed_by_user": False, "warnings": []},
    ]
    for extra in cases:
        doc = _v2_doc({"a.jpg": datetime(2026, 7, 23, 10, 15, 30)})
        doc.update(extra)
        timestamp_ocr.write_sidecar_doc(str(tmp_path), doc)
        _, template, values = timestamp_ocr.describe_sidecar(str(tmp_path))
        assert set(_re.findall(r"\{(\w+)\}", template)) == set(values), (template, values)
        template.format(**values)  # must not raise

def test_unknown_version_still_yields_frames(tmp_path):
    doc = _v2_doc({"a.jpg": datetime(2026, 7, 23, 10, 15, 30)})
    doc["version"] = 99
    timestamp_ocr.write_sidecar_doc(str(tmp_path), doc)
    assert "a.jpg" in timestamp_ocr.load_sidecar(str(tmp_path))


def test_sidecar_doc_none_for_non_object_json(tmp_path):
    (tmp_path / timestamp_ocr.config.TIMESTAMP_SIDECAR_NAME).write_text("[1, 2, 3]")
    assert timestamp_ocr.load_sidecar_doc(str(tmp_path)) is None
    assert timestamp_ocr.load_sidecar(str(tmp_path)) == {}


def test_write_sidecar_leaves_no_temp_file(tmp_path):
    timestamp_ocr.write_sidecar_doc(str(tmp_path), _v2_doc({}))
    assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())


# --- scan_folder --------------------------------------------------------------

def test_scan_folder_samples_and_fits(tmp_path):
    cv2 = pytest.importorskip("cv2")

    # 60 frames of a 25fps clip sampled every 25 source frames (1s apart).
    names = []
    for i in range(60):
        name = f"A_20260723_100000_{i * 25:06d}.jpg"
        cv2.imwrite(str(tmp_path / name), _blank_image())
        names.append(name)

    # The fake reader answers every call with the clock for whatever frame is
    # being read; it can't know which, so return a fixed advancing sequence.
    class _Clock:
        def __init__(self):
            self.calls = 0

        def readtext(self, image, detail=1):
            # First call is the region probe; then one per sample, 1s apart.
            second = max(0, self.calls - 1)
            text = f"2026-07-23 10:{second // 60:02d}:{second % 60:02d}"
            self.calls += 1
            result = [(_BBOX, text, 0.9)]
            return [t for _b, t, _c in result] if detail == 0 else result

    result = timestamp_ocr.scan_folder(str(tmp_path), names, reader=_Clock())

    assert result.x_kind == "frame_index"
    assert len(result.samples) < len(names)  # sampled, not exhaustive
    assert result.fit.ok
    assert set(result.crops) == {s.name for s in result.samples}
    assert result.unparsed == []


def test_scan_folder_skips_unreadable_images(tmp_path):
    pytest.importorskip("cv2")
    (tmp_path / "A_20260723_100000_000000.jpg").write_bytes(b"")
    (tmp_path / "A_20260723_100000_000025.jpg").write_bytes(b"")
    names = sorted(p.name for p in tmp_path.iterdir())
    result = timestamp_ocr.scan_folder(str(tmp_path), names, reader=_FakeReader([]))
    assert result.samples == []
    assert result.fit.ok is False


# --- scan_folder: region_override ----------------------------------------

def test_scan_folder_region_override_skips_the_probe(tmp_path):
    cv2 = pytest.importorskip("cv2")

    names = []
    for i in range(40):
        name = f"A_20260723_100000_{i * 25:06d}.jpg"
        cv2.imwrite(str(tmp_path / name), _blank_image())
        names.append(name)

    class _CountingReader:
        """Fails if asked to OCR anything but the tiny region-override crop."""

        def __init__(self):
            self.calls = 0

        def readtext(self, image, detail=1):
            self.calls += 1
            # A region_override crop is 30x15; the full frame is 200x40 --
            # this is how the test proves no full-frame probe happened.
            assert image.shape[:2] == (15, 30)
            result = [(_BBOX, "2026-07-23 10:15:30", 0.9)]
            return [t for _b, t, _c in result] if detail == 0 else result

    reader = _CountingReader()
    result = timestamp_ocr.scan_folder(str(tmp_path), names, reader=reader,
                                       region_override=(0, 0, 30, 15))
    assert result.region == (0, 0, 30, 15)
    assert result.region_is_manual is True
    assert reader.calls == len(result.samples)  # one call per sample, no probe


def test_scan_folder_without_override_is_not_manual(tmp_path):
    cv2 = pytest.importorskip("cv2")
    name = "A_20260723_100000_000000.jpg"
    cv2.imwrite(str(tmp_path / name), _blank_image())
    reader = _FakeReader([[(_BBOX, "2026-07-23 10:15:30", 0.9)]])
    result = timestamp_ocr.scan_folder(str(tmp_path), [name], reader=reader)
    assert result.region_is_manual is False


# --- load_scan_for_review: reopen a saved review without OCR ----------------

def _write_reviewable_doc(tmp_path, names, region=(1, 2, 3, 4)):
    """A v2 sidecar shaped like one TimelineReviewDialog.Apply would write."""
    doc = {
        "version": 2, "source": "timeline", "x_kind": "frame_index",
        "samples": [
            {"name": name, "clip": 0, "x": i * 25,
             "chosen_time": f"2026-07-23T10:{i:02d}:00",
             "filename_time": "2026-07-23T10:00:00",
             "edited": False, "ignored": False,
             "candidates": [{"text": f"2026-07-23 10:{i:02d}:00",
                            "time": f"2026-07-23T10:{i:02d}:00", "conf": 0.9}]}
            for i, name in enumerate(names)
        ],
        "frames": {name: f"2026-07-23T10:{i:02d}:00" for i, name in enumerate(names)},
        "region": list(region) if region else None,
        "region_is_manual": region is not None,
    }
    timestamp_ocr.write_sidecar_doc(str(tmp_path), doc)
    return doc


def test_load_scan_for_review_reconstructs_samples_without_ocr(tmp_path):
    cv2 = pytest.importorskip("cv2")
    names = [f"A_20260723_100000_{i * 25:06d}.jpg" for i in range(5)]
    for name in names:
        cv2.imwrite(str(tmp_path / name), _blank_image())
    _write_reviewable_doc(tmp_path, names)

    result = timestamp_ocr.load_scan_for_review(str(tmp_path))
    assert result is not None
    assert len(result.samples) == 5
    assert result.region == (1, 2, 3, 4)
    assert result.region_is_manual is True
    assert result.samples[0].chosen == datetime(2026, 7, 23, 10, 0, 0)
    assert set(result.crops) == {s.name for s in result.samples}


def test_load_scan_for_review_none_without_a_sidecar(tmp_path):
    assert timestamp_ocr.load_scan_for_review(str(tmp_path)) is None


def test_load_scan_for_review_none_for_v1_sidecar(tmp_path):
    # A plain ocr_folder sidecar has no samples/fit to reopen into a review.
    timestamp_ocr._save_sidecar(str(tmp_path), {"a.jpg": datetime(2026, 7, 23, 10, 0, 0)})
    assert timestamp_ocr.load_scan_for_review(str(tmp_path)) is None


def test_load_scan_for_review_none_when_v2_has_no_samples(tmp_path):
    timestamp_ocr.write_sidecar_doc(str(tmp_path), {
        "version": 2, "source": "timeline", "frames": {}})
    assert timestamp_ocr.load_scan_for_review(str(tmp_path)) is None


def test_load_scan_for_review_survives_missing_frame_files(tmp_path):
    # The sidecar references frames; none of them exist on disk any more.
    # This reaches the crop-loading loop (unlike the "none returned" tests
    # above), so it does need cv2 importable, even though every read fails.
    pytest.importorskip("cv2")
    _write_reviewable_doc(tmp_path, ["gone1.jpg", "gone2.jpg"])
    result = timestamp_ocr.load_scan_for_review(str(tmp_path))
    assert result is not None
    assert result.crops == {}  # nothing to crop, but it doesn't raise


def test_load_scan_for_review_recomputes_the_fit(tmp_path):
    cv2 = pytest.importorskip("cv2")
    names = [f"A_20260723_100000_{i * 25:06d}.jpg" for i in range(20)]
    for name in names:
        cv2.imwrite(str(tmp_path / name), _blank_image())
    _write_reviewable_doc(tmp_path, names)

    result = timestamp_ocr.load_scan_for_review(str(tmp_path))
    assert result.fit.n_samples == 20
    assert result.fit.ok


def test_load_scan_for_review_preserves_manual_edits(tmp_path):
    cv2 = pytest.importorskip("cv2")
    names = [f"A_20260723_100000_{i * 25:06d}.jpg" for i in range(5)]
    for name in names:
        cv2.imwrite(str(tmp_path / name), _blank_image())
    doc = _write_reviewable_doc(tmp_path, names)
    doc["samples"][2]["edited"] = True
    doc["samples"][2]["chosen_time"] = "2026-07-23T11:00:00"
    timestamp_ocr.write_sidecar_doc(str(tmp_path), doc)

    result = timestamp_ocr.load_scan_for_review(str(tmp_path))
    edited = next(s for s in result.samples if s.name == names[2])
    assert edited.edited is True
    assert edited.chosen == datetime(2026, 7, 23, 11, 0, 0)


# --- built-in presets compile and parse -------------------------------------

def test_all_builtin_patterns_compile():
    import config
    for label, pattern in config.BUILTIN_OCR_TIME_PATTERNS:
        timestamp_ocr.compile_custom_pattern(pattern)  # must not raise


def test_builtin_iso_dot_pattern_matches_the_reported_format():
    import config
    patterns = dict(config.BUILTIN_OCR_TIME_PATTERNS)
    compiled = timestamp_ocr.compile_custom_pattern(patterns["ISO date, dot time"])
    ts = timestamp_ocr.parse_overlay_text("2026-07-24 17.40.50", compiled)
    assert ts == datetime(2026, 7, 24, 17, 40, 50)


def test_builtin_compact_pattern_parses_with_no_separators():
    import config
    patterns = dict(config.BUILTIN_OCR_TIME_PATTERNS)
    compiled = timestamp_ocr.compile_custom_pattern(patterns["Compact, no separators"])
    ts = timestamp_ocr.parse_overlay_text("20260724174050", compiled)
    assert ts == datetime(2026, 7, 24, 17, 40, 50)


def test_builtin_12_hour_pattern_applies_pm():
    import config
    patterns = dict(config.BUILTIN_OCR_TIME_PATTERNS)
    compiled = timestamp_ocr.compile_custom_pattern(patterns["Day/Month/Year, 12-hour clock"])
    ts = timestamp_ocr.parse_overlay_text("24/07/2026 05:40:50 PM", compiled)
    assert ts == datetime(2026, 7, 24, 17, 40, 50)


# --- saved patterns persistence ----------------------------------------------

def test_saved_patterns_round_trip(tmp_path):
    path = str(tmp_path / "ocr_patterns.json")
    timestamp_ocr.save_patterns({"My camera": "YYYY-MO-DD HH.MI.SS"}, path)
    assert timestamp_ocr.load_saved_patterns(path) == {"My camera": "YYYY-MO-DD HH.MI.SS"}


def test_saved_patterns_missing_file_returns_empty(tmp_path):
    assert timestamp_ocr.load_saved_patterns(str(tmp_path / "nope.json")) == {}


def test_saved_patterns_corrupt_file_returns_empty(tmp_path):
    path = tmp_path / "ocr_patterns.json"
    path.write_text("not json")
    assert timestamp_ocr.load_saved_patterns(str(path)) == {}


def test_saved_patterns_non_object_json_returns_empty(tmp_path):
    path = tmp_path / "ocr_patterns.json"
    path.write_text("[1, 2, 3]")
    assert timestamp_ocr.load_saved_patterns(str(path)) == {}


def test_saved_patterns_drops_non_string_values(tmp_path):
    path = tmp_path / "ocr_patterns.json"
    path.write_text('{"good": "YYYY-MO-DD HH:MI:SS", "bad": 123}')
    assert timestamp_ocr.load_saved_patterns(str(path)) == {"good": "YYYY-MO-DD HH:MI:SS"}


def test_save_patterns_failure_does_not_raise(tmp_path):
    bad_path = str(tmp_path / "no_such_dir" / "ocr_patterns.json")
    timestamp_ocr.save_patterns({"x": "YYYY-MO-DD HH:MI:SS"}, bad_path)  # must not raise
    assert not os.path.exists(bad_path)


def test_default_patterns_path_is_next_to_config():
    import config
    path = timestamp_ocr.default_patterns_path()
    assert os.path.basename(path) == "ocr_patterns.json"
    assert os.path.dirname(path) == os.path.dirname(os.path.abspath(config.__file__))


# --- pattern label stripping / choice listing -------------------------------

def test_strip_pattern_label_removes_annotation():
    assert timestamp_ocr.strip_pattern_label(
        "YYYY-MO-DD HH:MI:SS   (ISO date, colon time)") == "YYYY-MO-DD HH:MI:SS"


def test_strip_pattern_label_leaves_bare_pattern_unchanged():
    assert timestamp_ocr.strip_pattern_label("YYYY-MO-DD HH.MI.SS") == "YYYY-MO-DD HH.MI.SS"


def test_strip_pattern_label_leaves_a_lone_trailing_paren_alone():
    # No "   (" separator -- not our annotation format, don't mangle it.
    assert timestamp_ocr.strip_pattern_label("weird)") == "weird)"


def test_strip_pattern_label_strips_surrounding_whitespace():
    assert timestamp_ocr.strip_pattern_label("  YYYY-MO-DD HH:MI:SS  ") == "YYYY-MO-DD HH:MI:SS"


def test_pattern_choices_includes_builtins(tmp_path, monkeypatch):
    monkeypatch.setattr(timestamp_ocr, "default_patterns_path", lambda: str(tmp_path / "p.json"))
    import config
    choices = timestamp_ocr.pattern_choices()
    assert len(choices) == len(config.BUILTIN_OCR_TIME_PATTERNS)
    for label, pattern in config.BUILTIN_OCR_TIME_PATTERNS:
        assert f"{pattern}   ({label})" in choices


def test_pattern_choices_includes_saved_after_builtins(tmp_path, monkeypatch):
    path = str(tmp_path / "p.json")
    monkeypatch.setattr(timestamp_ocr, "default_patterns_path", lambda: path)
    import config
    timestamp_ocr.save_patterns({"My camera": "YYYY-MO-DD HH.MI.SS"}, path)
    choices = timestamp_ocr.pattern_choices()
    assert choices[-1] == "YYYY-MO-DD HH.MI.SS   (My camera)"
    assert len(choices) == len(config.BUILTIN_OCR_TIME_PATTERNS) + 1


def test_pattern_choices_round_trip_through_strip(tmp_path, monkeypatch):
    path = str(tmp_path / "p.json")
    monkeypatch.setattr(timestamp_ocr, "default_patterns_path", lambda: path)
    timestamp_ocr.save_patterns({"My camera": "YYYY-MO-DD HH.MI.SS"}, path)
    choice = timestamp_ocr.pattern_choices()[-1]
    bare = timestamp_ocr.strip_pattern_label(choice)
    timestamp_ocr.compile_custom_pattern(bare)  # must not raise
    assert bare == "YYYY-MO-DD HH.MI.SS"
