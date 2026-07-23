"""Central configuration and default values for the vehicle Re-ID pipeline.

Everything the GUI/CLI can tune lives here so behaviour is easy to reason about
in one place. Values are plain module-level constants and a small dataclass so
they can be imported without side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --- Detection -------------------------------------------------------------

# COCO class ids that count as "vehicle" for YOLO.
#   2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASS_IDS: tuple[int, ...] = (2, 3, 5, 7)

# Default detection model. Kept in sync with ``model_registry.DEFAULT_KEY``;
# yolov8n is small (~6MB) and auto-downloads on first use. The full list of
# selectable models lives in ``src/mash_reid/model_registry.py`` and is managed
# (download / update) via ``src/mash_reid/model_manager.py``.
YOLO_WEIGHTS = "yolov8n.pt"

# Minimum detection confidence to keep a vehicle box.
DEFAULT_DETECTION_CONF = 0.35

# Discard boxes smaller than this (pixels, area of the crop). Tiny far-away
# vehicles produce unreliable embeddings.
MIN_BOX_AREA = 24 * 24


# --- Timestamp parsing -----------------------------------------------------

# Ordered list of (regex, strptime-format) pairs tried against the filename.
# The first capturing group of the regex is fed to datetime.strptime.
# Extend this list to support your own naming convention.
TIMESTAMP_PATTERNS: list[tuple[str, str]] = [
    # A_20260723_101530.jpg  /  cam1-20260723_101530.png
    (r"(\d{8}_\d{6})", "%Y%m%d_%H%M%S"),
    # 2026-07-23_10-15-30.jpg
    (r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", "%Y-%m-%d_%H-%M-%S"),
    # 2026-07-23T10:15:30
    (r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", "%Y-%m-%dT%H:%M:%S"),
    # plain unix-ish 14-digit: 20260723101530
    (r"(\d{14})", "%Y%m%d%H%M%S"),
]

IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --- Matching --------------------------------------------------------------

# Cosine-similarity cutoff below which an A/B pair is never considered a match.
DEFAULT_SIMILARITY_THRESHOLD = 0.6

# How many best B-candidates to keep per A-vehicle.
DEFAULT_TOP_K = 5


@dataclass
class MatchConfig:
    """Runtime-tunable matching parameters (exposed via GUI sliders)."""

    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    top_k: int = DEFAULT_TOP_K
    # Temporal gate: a B sighting is only a candidate when
    #   min_travel_seconds <= (t_B - t_A) <= max_travel_seconds.
    # Set use_time_gate=False to ignore timestamps entirely.
    use_time_gate: bool = True
    min_travel_seconds: float = 0.0
    max_travel_seconds: float = 600.0  # 10 minutes
    # When True, force a one-to-one assignment across A and B (Hungarian).
    one_to_one: bool = False


@dataclass
class PipelineConfig:
    """Configuration for detection + embedding over a folder of frames."""

    yolo_weights: str = YOLO_WEIGHTS  # catalog key or a custom .pt path
    detection_conf: float = DEFAULT_DETECTION_CONF
    vehicle_class_ids: tuple[int, ...] = field(default_factory=lambda: VEHICLE_CLASS_IDS)
    min_box_area: int = MIN_BOX_AREA
    device: str | None = None  # None -> auto (cuda if available else cpu)
    # Folder for downloaded weights. None -> model_manager.default_models_dir()
    # (``<project>/models`` or $MASH_MODELS_DIR).
    models_dir: str | None = None
