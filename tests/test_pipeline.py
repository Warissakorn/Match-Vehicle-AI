"""Tests for pipeline helpers.

Covers the Unicode-safe image read that fixes "detection finds no vehicles"
when frames live in a non-ASCII (e.g. Thai) folder. cv2-gated so it is skipped
where OpenCV isn't installed (e.g. CI).
"""

import os

import pytest

from mash_reid import pipeline


def test_imread_unicode_reads_from_thai_path(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    # A real JPEG inside a Thai-named folder — the case where cv2.imread returns
    # None on Windows, making detection silently skip every frame.
    thai_dir = tmp_path / "กล้องจุดA"
    thai_dir.mkdir()
    img_path = str(thai_dir / "A_20260723_101500_000000.jpg")
    ok, buf = cv2.imencode(".jpg", np.full((48, 64, 3), 90, np.uint8))
    assert ok
    with open(img_path, "wb") as fh:
        fh.write(buf.tobytes())

    img = pipeline._imread_unicode(cv2, img_path)
    assert img is not None
    assert img.shape == (48, 64, 3)


def test_imread_unicode_missing_file_returns_none(tmp_path):
    cv2 = pytest.importorskip("cv2")
    assert pipeline._imread_unicode(cv2, str(tmp_path / "nope.jpg")) is None
