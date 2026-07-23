"""Export user-confirmed/rejected A/B matches as labeled training data.

When a user reviews a proposed match in the GUI and marks it "same vehicle" or
"different vehicle", this saves both vehicle crops plus a manifest row so the
labeled pairs can later be used to train or fine-tune a Re-ID model.

Layout under ``output_dir``:

    positive/<pair_id>_A.jpg   # label = same vehicle
    positive/<pair_id>_B.jpg
    negative/<pair_id>_A.jpg   # label = different vehicle
    negative/<pair_id>_B.jpg
    manifest.csv               # one row per exported pair
"""

from __future__ import annotations

import csv
import logging
import os
import uuid
from datetime import datetime

from mash_reid.image_io import imread_unicode, imwrite_unicode
from mash_reid.matcher import VehicleRecord

log = logging.getLogger(__name__)

_MANIFEST_NAME = "manifest.csv"
_MANIFEST_FIELDS = [
    "pair_id", "label", "similarity",
    "point_a_frame", "point_a_bbox", "point_b_frame", "point_b_bbox",
    "crop_a_path", "crop_b_path",
    "timestamp_a", "timestamp_b", "created_at",
]


def _crop(cv2, frame_path: str, bbox: tuple[int, int, int, int]):
    image = imread_unicode(cv2, frame_path)
    if image is None:
        raise RuntimeError(f"Could not read frame for crop: {frame_path}")
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2]


def export_labeled_pair(
    output_dir: str,
    rec_a: VehicleRecord,
    rec_b: VehicleRecord,
    label: bool,
    similarity: float,
    pair_id: str | None = None,
) -> str:
    """Save one labeled A/B pair for training. Returns the pair id used.

    ``label=True`` means the user confirmed these are the same vehicle
    (saved under ``positive/``); ``label=False`` means rejected
    (``negative/``).
    """
    import cv2  # deferred so importing this module stays OpenCV-free

    pair_id = pair_id or f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
    subdir = "positive" if label else "negative"
    label_dir = os.path.join(output_dir, subdir)
    os.makedirs(label_dir, exist_ok=True)

    crop_a = _crop(cv2, rec_a.frame_path, rec_a.bbox)
    crop_b = _crop(cv2, rec_b.frame_path, rec_b.bbox)

    crop_a_path = os.path.join(label_dir, f"{pair_id}_A.jpg")
    crop_b_path = os.path.join(label_dir, f"{pair_id}_B.jpg")
    imwrite_unicode(cv2, crop_a_path, crop_a, "jpg")
    imwrite_unicode(cv2, crop_b_path, crop_b, "jpg")

    _append_manifest_row(output_dir, {
        "pair_id": pair_id,
        "label": int(label),
        "similarity": f"{similarity:.4f}",
        "point_a_frame": rec_a.frame_path,
        "point_a_bbox": rec_a.bbox,
        "point_b_frame": rec_b.frame_path,
        "point_b_bbox": rec_b.bbox,
        "crop_a_path": crop_a_path,
        "crop_b_path": crop_b_path,
        "timestamp_a": rec_a.timestamp.isoformat(),
        "timestamp_b": rec_b.timestamp.isoformat(),
        "created_at": datetime.now().isoformat(),
    })

    log.info("Exported %s training pair %s (sim=%.3f) to %s",
             "positive" if label else "negative", pair_id, similarity, label_dir)
    return pair_id


def _append_manifest_row(output_dir: str, row: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, _MANIFEST_NAME)
    write_header = not os.path.exists(manifest_path)
    with open(manifest_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def count_labeled_pairs(output_dir: str) -> tuple[int, int]:
    """Return (positive_count, negative_count) already exported, for GUI status."""
    def _count(sub):
        d = os.path.join(output_dir, sub)
        if not os.path.isdir(d):
            return 0
        return len([f for f in os.listdir(d) if f.endswith("_A.jpg")])
    return _count("positive"), _count("negative")
