"""Tests for exporting confirm/reject-labeled A/B pairs as training data.

cv2-gated (skipped where OpenCV isn't installed, e.g. CI) since crops require
real image encode/decode.
"""

import csv
import os
from datetime import datetime

import pytest

from mash_reid import training_export as te
from mash_reid.matcher import VehicleRecord


def _write_frame(cv2, np, path, size=(64, 48), fill=100):
    ok, buf = cv2.imencode(".jpg", np.full((size[1], size[0], 3), fill, np.uint8))
    assert ok
    with open(path, "wb") as fh:
        fh.write(buf.tobytes())


def _rec(cv2, np, tmp_path, name, point, embedding_fill=100):
    frame_path = str(tmp_path / name)
    _write_frame(cv2, np, frame_path, fill=embedding_fill)
    return VehicleRecord(
        record_id=0, point=point, frame_path=frame_path,
        timestamp=datetime(2026, 7, 23, 10, 0, 0),
        bbox=(4, 4, 40, 30), confidence=0.9,
        embedding=np.zeros(4, dtype="float32"),
    )


def test_export_positive_pair_writes_crops_and_manifest(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    out_dir = str(tmp_path / "training_data")
    rec_a = _rec(cv2, np, tmp_path, "A_frame.jpg", "A")
    rec_b = _rec(cv2, np, tmp_path, "B_frame.jpg", "B")

    pair_id = te.export_labeled_pair(out_dir, rec_a, rec_b, label=True, similarity=0.87)

    crop_a = os.path.join(out_dir, "positive", f"{pair_id}_A.jpg")
    crop_b = os.path.join(out_dir, "positive", f"{pair_id}_B.jpg")
    assert os.path.getsize(crop_a) > 0
    assert os.path.getsize(crop_b) > 0

    manifest_path = os.path.join(out_dir, "manifest.csv")
    with open(manifest_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["pair_id"] == pair_id
    assert rows[0]["label"] == "1"
    assert rows[0]["similarity"] == "0.8700"


def test_export_negative_pair_goes_to_negative_dir(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    out_dir = str(tmp_path / "training_data")
    rec_a = _rec(cv2, np, tmp_path, "A_frame.jpg", "A")
    rec_b = _rec(cv2, np, tmp_path, "B_frame.jpg", "B")

    pair_id = te.export_labeled_pair(out_dir, rec_a, rec_b, label=False, similarity=0.42)

    assert os.path.isfile(os.path.join(out_dir, "negative", f"{pair_id}_A.jpg"))
    assert not os.path.isdir(os.path.join(out_dir, "positive"))


def test_manifest_appends_across_multiple_exports(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    out_dir = str(tmp_path / "training_data")
    rec_a = _rec(cv2, np, tmp_path, "A_frame.jpg", "A")
    rec_b = _rec(cv2, np, tmp_path, "B_frame.jpg", "B")

    te.export_labeled_pair(out_dir, rec_a, rec_b, label=True, similarity=0.9)
    te.export_labeled_pair(out_dir, rec_a, rec_b, label=False, similarity=0.3)

    with open(os.path.join(out_dir, "manifest.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2


def test_count_labeled_pairs(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    out_dir = str(tmp_path / "training_data")
    assert te.count_labeled_pairs(out_dir) == (0, 0)

    rec_a = _rec(cv2, np, tmp_path, "A_frame.jpg", "A")
    rec_b = _rec(cv2, np, tmp_path, "B_frame.jpg", "B")
    te.export_labeled_pair(out_dir, rec_a, rec_b, label=True, similarity=0.9)
    te.export_labeled_pair(out_dir, rec_a, rec_b, label=False, similarity=0.2)
    te.export_labeled_pair(out_dir, rec_a, rec_b, label=False, similarity=0.1)

    assert te.count_labeled_pairs(out_dir) == (1, 2)
