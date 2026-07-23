"""Orchestrate: a folder of frames -> a list of embedded ``VehicleRecord``.

This ties frame_loader -> detector -> embedder together and hands the result to
``matcher.match``. Detection + embedding are the expensive steps, so results are
optionally cached to disk keyed by (folder contents + config), letting the GUI
re-match with new thresholds instantly without re-running the models.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import dataclass

import numpy as np

import config
from mash_reid import frame_loader
from mash_reid.detector import VehicleDetector
from mash_reid.embedder import Embedder, get_default_embedder
from mash_reid.matcher import VehicleRecord


@dataclass
class PointResult:
    """All vehicles detected+embedded for one camera point."""

    point: str
    folder: str
    records: list[VehicleRecord]
    frame_count: int


def _imread_unicode(cv2, path: str):
    """Read an image, supporting non-ASCII paths; returns a BGR array or None.

    ``cv2.imread`` returns None (no error) for paths with non-ASCII characters
    (e.g. Thai) on Windows, which made detection silently find nothing when
    frames lived in such a folder. Reading the bytes with Python's ``open`` and
    decoding via ``cv2.imdecode`` handles Unicode paths on every platform.
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _cache_key(folder: str, cfg: config.PipelineConfig) -> str:
    """Stable hash over filenames+mtimes+sizes and the detection config."""
    h = hashlib.sha256()
    h.update(repr((cfg.yolo_weights, cfg.detection_conf,
                   tuple(cfg.vehicle_class_ids), cfg.min_box_area)).encode())
    for entry in sorted(os.listdir(folder)):
        ext = os.path.splitext(entry)[1].lower()
        if ext not in config.IMAGE_EXTENSIONS:
            continue
        p = os.path.join(folder, entry)
        st = os.stat(p)
        h.update(f"{entry}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def process_point(
    folder: str,
    point: str,
    detector: VehicleDetector,
    embedder: Embedder,
    cfg: config.PipelineConfig | None = None,
    use_cache: bool = True,
    progress=None,
) -> PointResult:
    """Detect + embed every vehicle in ``folder`` for camera ``point``.

    ``progress`` is an optional callable ``(done, total, message)`` for GUIs.
    """
    import cv2  # deferred so importing the module stays light

    cfg = cfg or config.PipelineConfig()
    frames = frame_loader.load_frames(folder, point)

    cache_path = None
    if use_cache:
        cache_path = os.path.join(folder, f".{_cache_key(folder, cfg)}.reidcache")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as fh:
                    records = pickle.load(fh)
                if progress:
                    progress(len(frames), len(frames), "loaded from cache")
                return PointResult(point, folder, records, len(frames))
            except Exception:
                pass  # corrupt cache -> recompute

    records: list[VehicleRecord] = []
    next_id = 0
    total = len(frames)
    for idx, frame in enumerate(frames):
        image = _imread_unicode(cv2, frame.path)
        if image is None:
            continue
        detections = detector.detect(image)
        crops = [d.crop for d in detections]
        embeddings = embedder.embed_batch(crops)
        for det, emb in zip(detections, embeddings):
            records.append(
                VehicleRecord(
                    record_id=next_id,
                    point=point,
                    frame_path=frame.path,
                    timestamp=frame.timestamp,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    embedding=emb.astype(np.float32),
                )
            )
            next_id += 1
        if progress:
            progress(idx + 1, total, frame.name)

    if cache_path:
        try:
            with open(cache_path, "wb") as fh:
                pickle.dump(records, fh)
        except Exception:
            pass

    return PointResult(point, folder, records, total)


def build_pipeline(cfg: config.PipelineConfig | None = None):
    """Convenience factory returning (detector, embedder) sharing one config."""
    cfg = cfg or config.PipelineConfig()
    detector = VehicleDetector(cfg)
    embedder = get_default_embedder(device=cfg.device)
    return detector, embedder
