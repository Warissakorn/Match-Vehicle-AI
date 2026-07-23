"""Catalog of selectable vehicle-detection models.

This is the single source of truth for *which* detection models the app offers
in its dropdown / CLI. It is pure data (no torch, no network, no file IO), so it
imports cheaply and is safe to read on startup.

Every entry is an Ultralytics YOLO checkpoint pretrained on COCO — so the
vehicle classes in ``config.VEHICLE_CLASS_IDS`` (car / motorcycle / bus / truck)
apply unchanged. Two families are offered:

* **YOLOv8** — the long-standing baseline the project shipped with.
* **YOLO11** — the newer generation; a little faster and more accurate at the
  same size. Marked ``recommended`` so the UI can nudge users toward it, which
  is how "keep the model up to date" is surfaced.

Within a family the suffix trades speed for accuracy: ``n`` (nano, fastest) →
``s`` → ``m`` → ``l`` → ``x`` (largest, most accurate, slowest).

To offer another model, add a ``ModelInfo`` below — nothing else needs to know.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """One selectable detection model."""

    key: str            # stable id / weights filename, e.g. "yolov8n.pt"
    family: str         # "YOLOv8" | "YOLO11"
    size: str           # "nano" | "small" | "medium" | "large" | "xlarge"
    approx_mb: float    # download size, for the UI (not enforced)
    description: str    # one-line speed/accuracy hint
    recommended: bool = False  # newer generation the UI should prefer

    @property
    def display_name(self) -> str:
        star = " ★" if self.recommended else ""
        return f"{self.family} {self.size} ({self.approx_mb:.0f} MB){star}"


# Ordered catalog — the display order in dropdowns / `models list`.
DETECTION_MODELS: tuple[ModelInfo, ...] = (
    # --- YOLO11: newer generation, recommended -----------------------------
    ModelInfo("yolo11n.pt", "YOLO11", "nano", 5.4,
              "Newest gen, fastest — best default for CPU / quick runs.", True),
    ModelInfo("yolo11s.pt", "YOLO11", "small", 18.4,
              "Newest gen, small — better accuracy, still fast.", True),
    ModelInfo("yolo11m.pt", "YOLO11", "medium", 38.8,
              "Newest gen, medium — balanced accuracy/speed.", True),
    ModelInfo("yolo11l.pt", "YOLO11", "large", 49.0,
              "Newest gen, large — high accuracy, needs more compute.", True),
    ModelInfo("yolo11x.pt", "YOLO11", "xlarge", 109.3,
              "Newest gen, largest — most accurate, slowest.", True),
    # --- YOLOv8: the original baseline --------------------------------------
    ModelInfo("yolov8n.pt", "YOLOv8", "nano", 6.2,
              "Baseline nano — the project's original default."),
    ModelInfo("yolov8s.pt", "YOLOv8", "small", 21.5,
              "Baseline small — a bit more accurate than nano."),
    ModelInfo("yolov8m.pt", "YOLOv8", "medium", 49.7,
              "Baseline medium — balanced accuracy/speed."),
    ModelInfo("yolov8l.pt", "YOLOv8", "large", 83.7,
              "Baseline large — high accuracy, heavier."),
    ModelInfo("yolov8x.pt", "YOLOv8", "xlarge", 130.5,
              "Baseline largest — most accurate of the v8 family."),
)

# Kept in sync with ``config.YOLO_WEIGHTS`` — the model used when none is chosen.
DEFAULT_KEY = "yolov8n.pt"

_BY_KEY = {m.key: m for m in DETECTION_MODELS}


def all_models() -> tuple[ModelInfo, ...]:
    """Every catalog entry, in display order."""
    return DETECTION_MODELS


def keys() -> list[str]:
    """All model keys, for CLI ``choices=`` and validation."""
    return [m.key for m in DETECTION_MODELS]


def is_known(key: str) -> bool:
    """True if ``key`` names a catalog model (vs. a custom weights path)."""
    return key in _BY_KEY


def get(key: str) -> ModelInfo:
    """Look up a model by key; raises ``KeyError`` with the valid options."""
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(
            f"Unknown detection model '{key}'. Available: {', '.join(keys())}"
        ) from None


def default() -> ModelInfo:
    """The catalog entry used when the user hasn't chosen one."""
    return _BY_KEY[DEFAULT_KEY]
