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
