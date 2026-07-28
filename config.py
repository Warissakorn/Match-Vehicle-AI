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

# Cosine-similarity cutoff for grouping repeat detections of the same vehicle
# WITHIN one point (e.g. a car circling back past the same camera). Higher than
# the cross-point threshold since same-point detections share the same camera
# angle/lighting, so genuine repeats look more alike than cross-point matches.
DEFAULT_SAME_POINT_SIMILARITY_THRESHOLD = 0.8

# Whether to group repeat sightings within a point at all. Off by default:
# clustering builds a full N x N self-similarity matrix per point, which for a
# real gallery (N ~ 12,000, dim 2048) is ~590 GFLOP and peaks over 1 GB per
# call -- and its result is not cached, so on a warm .reidcache it becomes the
# single dominant cost of a run. Nothing downstream depends on it: matching,
# the time gate, and the training export are all independent, and the only
# consumer is a "*GrpN(xK)" suffix on gallery captions. Turn it back on with
# the GUI checkbox or the CLI's --cluster-same-point when the grouping is
# actually wanted.
DEFAULT_ENABLE_SAME_POINT_CLUSTERING = False

# Folder for confirm/reject-labeled training pairs exported from the GUI.
DEFAULT_TRAINING_DATA_DIR = "training_data"

# Real galleries can hold thousands of vehicles per point. Each thumbnail
# card loads an image from disk and creates several Tkinter widgets, so
# rendering every single one freezes the GUI for real-world frame counts.
# Gallery views cap at this many cards (earliest by time) and say so in the
# status/label text; narrowing by selecting an A-vehicle (top-k candidates
# only) is unaffected.
DEFAULT_MAX_GALLERY_THUMBNAILS = 300


# --- Timestamp OCR -----------------------------------------------------------

# Filename of the per-folder cache written by ``mash_reid.timestamp_ocr``.
# ``frame_loader`` reads it (when present) ahead of the filename/EXIF/mtime
# fallbacks, since it reflects the clock actually burned into the footage.
TIMESTAMP_SIDECAR_NAME = ".timestamps.json"

# Languages passed to easyocr.Reader -- overlays here are just digits and
# separators, so English is sufficient (and keeps the model download small).
OCR_LANGUAGES: tuple[str, ...] = ("en",)

# Padding (x, y) added around the detected overlay bounding box before
# cropping, so a slightly tighter per-frame box still contains the whole text.
OCR_REGION_PAD: tuple[int, int] = (10, 6)


# --- Timeline fit -------------------------------------------------------------
#
# A CCTV clock advances linearly with the source frame index, so instead of
# OCR-ing every frame we OCR a sample and fit a line through it. See
# ``mash_reid.timeline``.

# How many frames to sample per folder, and the floor per clip. The target is
# a floor rather than a cap: with a dozen clips, 40 samples spread evenly
# would leave 3 per clip, too few to place a clip's start time reliably.
DEFAULT_TIMELINE_SAMPLE_TARGET = 40
DEFAULT_TIMELINE_MIN_SAMPLES_PER_CLIP = 8

# Robust-fit tuning. A reading is an outlier when its residual exceeds
# ``SIGMA * 1.4826 * MAD`` -- but the overlay clock only shows whole seconds,
# so on clean footage the MAD can be exactly 0, which would make the tolerance
# 0 and reject every frame that isn't bit-exact. The floor (just over the 1 s
# display quantum) is what keeps that from happening.
TIMELINE_INLIER_SIGMA = 3.0
TIMELINE_INLIER_FLOOR_SECONDS = 1.5

# The overlay shows whole seconds (truncated), so readings sit on average half
# a second behind the true time; added back when the fit is applied.
TIMELINE_CLOCK_QUANTUM_SECONDS = 1.0

# Thresholds that turn into human-readable warnings in the review dialog. None
# of these block anything -- the user decides -- except a non-positive slope,
# which means time runs backwards and can only be a broken fit.
TIMELINE_MIN_INLIER_FRACTION = 0.6
TIMELINE_WARN_RESIDUAL_MAD_SECONDS = 2.0
TIMELINE_WARN_MAX_RESIDUAL_SECONDS = 10.0
TIMELINE_PLAUSIBLE_FPS_RANGE: tuple[float, float] = (1.0, 240.0)
TIMELINE_MIN_SAMPLES_FOR_CONFIDENT_CLIP = 3


# --- GPU / device ------------------------------------------------------------

# Cap on how many crops go through the embedder in a single forward pass.
# The embedder used to batch every detection in a frame with no limit, which
# risks a CUDA out-of-memory error on a busy frame; chunking bounds peak
# GPU memory use regardless of how many vehicles one frame contains.
DEFAULT_EMBED_BATCH_SIZE = 64

# Run the embedder in fp16 on CUDA (roughly halves VRAM use and is faster on
# most GPUs); ignored on CPU, where half precision isn't a win.
USE_HALF_PRECISION_ON_CUDA = True


# --- Resource monitor ---------------------------------------------------------

# How often the GUI polls CPU/RAM/GPU usage for the resource bar (ms).
RESOURCE_POLL_MS = 1000


# --- Settings persistence -----------------------------------------------------

# Where the GUI remembers folder paths, sliders, and other choices between
# runs. Lives at the project root, next to config.py, and is git-ignored.
DEFAULT_SETTINGS_FILE = "settings.json"

# Where user-saved custom OCR time patterns are kept (see
# mash_reid.timestamp_ocr.load_saved_patterns/save_patterns), separate from
# settings.json since it's a small named collection rather than a single
# flat value per key.
DEFAULT_OCR_PATTERNS_FILE = "ocr_patterns.json"

# Ready-made (label, pattern) choices covering the overlay layouts reported
# in practice, offered in the timestamp review dialog's pattern picker so
# most users never need to write a token template themselves -- see
# mash_reid.timestamp_ocr.compile_custom_pattern for the token vocabulary.
BUILTIN_OCR_TIME_PATTERNS: list[tuple[str, str]] = [
    ("ISO date, colon time", "YYYY-MO-DD HH:MI:SS"),
    ("ISO date, dot time", "YYYY-MO-DD HH.MI.SS"),
    ("Day/Month/Year, colon time", "DD/MO/YYYY HH:MI:SS"),
    ("Day/Month/Year, 12-hour clock", "DD/MO/YYYY hh:MI:SS AP"),
    ("Month/Day/Year, colon time", "MO/DD/YYYY HH:MI:SS"),
    ("Compact, no separators", "YYYYMODDHHMISS"),
]


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
    # Max crops per embedder forward pass -- see DEFAULT_EMBED_BATCH_SIZE.
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE
