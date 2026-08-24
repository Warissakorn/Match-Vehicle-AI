"""Orchestrate: a folder of frames -> a list of embedded ``VehicleRecord``.

This ties frame_loader -> detector -> embedder together and hands the result to
``matcher.match``. Detection + embedding are the expensive steps, so results are
optionally cached to disk keyed by (folder contents + config), letting the GUI
re-match with new thresholds instantly without re-running the models.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import threading
from dataclasses import dataclass

import numpy as np

import config
from mash_reid import frame_loader
from mash_reid.detector import VehicleDetector
from mash_reid.embedder import Embedder, get_default_embedder
from mash_reid.image_io import imread_unicode as _imread_unicode
from mash_reid.matcher import VehicleRecord

log = logging.getLogger(__name__)


@dataclass
class PointResult:
    """All vehicles detected+embedded for one camera point."""

    point: str
    folder: str
    records: list[VehicleRecord]
    frame_count: int


def _cache_key(folder: str, cfg: config.PipelineConfig) -> str:
    """Stable hash over filenames+mtimes+sizes and the detection config.

    Includes the OCR timestamp sidecar's own mtime+size (when present) so
    that running "Fix times (OCR)" -- which rewrites timestamps but not the
    frames themselves -- invalidates the cache instead of silently keeping
    the pre-OCR (wrong) timestamps. The version prefix guards the reverse
    case: caches written before ``VehicleRecord.timestamp_source`` existed
    are also treated as stale rather than unpickled into the new shape.
    Bumped to v3 when the sampled-timeline fit added the "timeline" source
    value, so old pickles don't carry a now-wrong label.
    """
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
    sidecar = os.path.join(folder, config.TIMESTAMP_SIDECAR_NAME)
    if os.path.exists(sidecar):
        st = os.stat(sidecar)
        h.update(f"ocr:{st.st_size}:{int(st.st_mtime)}".encode())
    return "v3." + h.hexdigest()[:16]


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
    log.info("Point %s: loaded %d frame(s) from %s", point, len(frames), folder)

    cache_path = None
    if use_cache:
        cache_path = os.path.join(folder, f".{_cache_key(folder, cfg)}.reidcache")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as fh:
                    records = pickle.load(fh)
                log.info("Point %s: loaded %d vehicle(s) from cache %s",
                         point, len(records), cache_path)
                if progress:
                    progress(len(frames), len(frames), "loaded from cache")
                return PointResult(point, folder, records, len(frames))
            except Exception:
                log.warning("Point %s: cache %s unreadable, recomputing",
                            point, cache_path, exc_info=True)

    records: list[VehicleRecord] = []
    next_id = 0
    total = len(frames)
    unreadable = 0
    for idx, frame in enumerate(frames):
        image = _imread_unicode(cv2, frame.path)
        if image is None:
            unreadable += 1
            log.warning("Point %s: could not read image %s (skipped)", point, frame.path)
            continue
        detections = detector.detect(image)
        log.debug("Point %s: %s -> %d vehicle(s)", point, frame.name, len(detections))
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
                    timestamp_source=frame.timestamp_source,
                )
            )
            next_id += 1
        if progress:
            progress(idx + 1, total, frame.name)

    log.info("Point %s: detected %d vehicle(s) across %d frame(s)%s",
             point, len(records), total,
             f", {unreadable} unreadable" if unreadable else "")
    if records == [] and unreadable == total and total > 0:
        log.warning("Point %s: every frame was unreadable — if the folder path "
                    "contains non-ASCII characters on Windows this was the cause; "
                    "it is now handled, so re-run.", point)

    if cache_path:
        try:
            with open(cache_path, "wb") as fh:
                pickle.dump(records, fh)
            log.debug("Point %s: cached %d vehicle(s) to %s", point, len(records), cache_path)
        except Exception:
            log.warning("Point %s: could not write cache %s", point, cache_path, exc_info=True)

    return PointResult(point, folder, records, total)


def build_pipeline(cfg: config.PipelineConfig | None = None):
    """Convenience factory returning (detector, embedder) sharing one config.

    One-shot callers (the CLI) want this. Long-lived callers (the GUI) want
    :func:`get_pipeline`, which keeps the models resident across runs.
    """
    cfg = cfg or config.PipelineConfig()
    detector = VehicleDetector(cfg)
    embedder = get_default_embedder(device=cfg.device, batch_size=cfg.embed_batch_size)
    return detector, embedder


# --- model residency cache -----------------------------------------------------
#
# The detector and embedder lazy-load their neural networks on first use and
# keep them loaded -- which is only a win if the same wrapper objects survive
# between runs. The GUI used to call ``build_pipeline`` on every Process
# click, throwing away fully-loaded models and re-loading YOLO + ResNet50
# (seconds each click, plus VRAM churn) for identical settings. One cached
# pair, keyed by everything that shapes construction, fixes that.

_pipeline_lock = threading.Lock()
_pipeline_key: tuple | None = None
_pipeline_value: tuple[VehicleDetector, Embedder] | None = None


def _pipeline_identity(cfg: config.PipelineConfig) -> tuple:
    """The config fields that decide how the models get built.

    ``detection_conf``, ``vehicle_class_ids`` and ``min_box_area`` are read
    per-call at detect time (see ``VehicleDetector.detect``), so changing them
    must NOT trigger a reload -- ``get_pipeline`` instead points the cached
    detector at the fresh config.
    """
    return (cfg.yolo_weights, cfg.device, cfg.embed_batch_size, cfg.models_dir)


def drop_pipeline() -> None:
    """Forget the resident models (next run rebuilds them from scratch)."""
    global _pipeline_key, _pipeline_value
    with _pipeline_lock:
        _pipeline_key = None
        _pipeline_value = None


def get_pipeline(cfg: config.PipelineConfig | None = None):
    """Return ``(detector, embedder)`` for ``cfg``, reusing loaded models.

    A cache hit returns the very objects from last time -- weights already on
    device, first-use cost already paid -- so re-running Process after only
    tweaking thresholds skips model loading entirely. Any change to the
    constructor-shaping fields (weights key, device, batch size, models dir)
    misses and rebuilds; per-call knobs are synced onto the cached detector so
    they always reflect the caller's current settings.
    """
    global _pipeline_key, _pipeline_value

    cfg = cfg or config.PipelineConfig()
    key = _pipeline_identity(cfg)
    with _pipeline_lock:
        if _pipeline_value is not None and _pipeline_key == key:
            detector, _ = _pipeline_value
            detector.cfg = cfg  # runtime knobs track the fresh config
            return _pipeline_value
        value = build_pipeline(cfg)
        _pipeline_key, _pipeline_value = key, value
        return value
