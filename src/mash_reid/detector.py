"""Vehicle detection with Ultralytics YOLO.

Given an image, return every vehicle box (car / motorcycle / bus / truck) as a
``Detection`` carrying its pixel box, the cropped BGR image, and confidence.
The cropped image is what the embedder turns into a Re-ID feature vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config


@dataclass
class Detection:
    """One detected vehicle within a frame."""

    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (pixels)
    crop: np.ndarray                 # BGR image of the vehicle
    confidence: float
    class_id: int

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


class VehicleDetector:
    """Thin wrapper around a YOLO model that keeps only vehicle classes."""

    def __init__(self, cfg: config.PipelineConfig | None = None):
        self.cfg = cfg or config.PipelineConfig()
        self._model = None  # lazy-loaded so import stays cheap / offline-safe

    def _ensure_model(self):
        if self._model is None:
            from ultralytics import YOLO  # heavy import, deferred

            self._model = YOLO(self.cfg.yolo_weights)
            if self.cfg.device:
                self._model.to(self.cfg.device)
        return self._model

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect vehicles in a single BGR image (as loaded by cv2.imread)."""
        model = self._ensure_model()
        results = model.predict(
            image,
            conf=self.cfg.detection_conf,
            classes=list(self.cfg.vehicle_class_ids),
            verbose=False,
        )

        detections: list[Detection] = []
        h, w = image.shape[:2]
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                # Clamp to image bounds.
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image[y1:y2, x1:x2].copy()
                det = Detection(
                    bbox=(x1, y1, x2, y2),
                    crop=crop,
                    confidence=float(box.conf[0]),
                    class_id=int(box.cls[0]),
                )
                if det.area < self.cfg.min_box_area:
                    continue
                detections.append(det)
        return detections
