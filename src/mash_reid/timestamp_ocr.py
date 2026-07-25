"""Read the date/time burned into a CCTV frame and cache it as a sidecar file.

Problem this solves: the pipeline's timestamps used to come from the video's
*filename* (start time) plus ``frame_index / fps`` (see ``video_extractor``).
That's only as accurate as the camera's declared fps and the filename's start
time -- in the field these drifted from the real on-screen clock, and since
the travel-time gate in ``matcher`` depends directly on timestamps, a wrong
clock means wrong (or missing) matches. Many CCTV feeds already burn the true
capture time into the frame as on-screen text; OCR-ing that text gives a
timestamp that matches the *actual* footage instead of an assumption about it.

Design: OCR the whole folder **once** and cache the result as
``<folder>/<config.TIMESTAMP_SIDECAR_NAME>`` (a small JSON file of
``{filename: iso_timestamp}``). ``frame_loader`` then reads that sidecar like
any other timestamp source -- OCR is a one-time cost per folder, not a
per-run cost, which matters at gallery sizes in the thousands.

Two layers, deliberately split so the hard part is unit-testable without any
of the heavy optional dependencies:

    * ``parse_overlay_text`` -- pure string -> datetime parsing, no OCR, no
      image I/O, no dependencies. This is where the accuracy actually lives
      (OCR reads noisy characters; this has to make sense of them).
    * ``get_reader`` / ``find_overlay_region`` / ``read_timestamp`` /
      ``ocr_folder`` -- the imaging side, built on ``easyocr`` + ``cv2``
      (both deferred imports, matching ``detector.py``/``embedder.py``).
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime

import config
from mash_reid.image_io import imread_unicode

log = logging.getLogger(__name__)

# --- pure text parsing -------------------------------------------------------

# OCR commonly misreads these characters as digits inside a date/time overlay
# (font-dependent, but these cover the vast majority of real CCTV overlays).
_DIGIT_FIX = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "l": "1", "I": "1", "|": "1", "i": "1",
    "S": "5", "s": "5",
    "B": "8",
    "Z": "2", "z": "2",
    "G": "6",
})

_D = "[0-9OoQDlI|iSsBZzG]"  # a single OCR-noisy "digit"
_SEP = r"[-/.\s]"


def _fix(s: str) -> str:
    return s.translate(_DIGIT_FIX)


def _valid_date(year: int, month: int, day: int) -> bool:
    return 2000 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31


def _resolve_two_digit_date(g1: int, g2: int, year: int) -> tuple[int, int, int] | None:
    """Disambiguate a DD/MM vs MM/DD overlay using validity as the tiebreak.

    Day-first (DD-MM-YYYY) is tried first since it's the common convention on
    Thai/EU CCTV overlays; month-first (MM-DD-YYYY, common in US exports) is
    the fallback when day-first is impossible (e.g. "07-23-2026" can't be
    day=07/month=23).
    """
    for day, month in ((g1, g2), (g2, g1)):
        if _valid_date(year, month, day):
            return year, month, day
    return None


# (regex, resolver) pairs, tried in order; first plausible date wins.
_ISO_DATE = re.compile(rf"({_D}{{4}}){_SEP}({_D}{{2}}){_SEP}({_D}{{2}})")
_TWO_DIGIT_DATE = re.compile(rf"({_D}{{2}}){_SEP}({_D}{{2}}){_SEP}({_D}{{4}})")
_COMPACT_DATE = re.compile(rf"({_D}{{4}})({_D}{{2}})({_D}{{2}})(?!{_D})")

_TIME_COLON = re.compile(rf"({_D}{{1,2}}):({_D}{{2}}):({_D}{{2}})\s*(AM|PM|am|pm)?")
_TIME_DASH = re.compile(rf"(?<!{_D})({_D}{{2}})-({_D}{{2}})-({_D}{{2}})\s*(AM|PM|am|pm)?")


def _find_date(text: str) -> tuple[int, int, int] | None:
    m = _ISO_DATE.search(text)
    if m:
        try:
            year, month, day = int(_fix(m.group(1))), int(_fix(m.group(2))), int(_fix(m.group(3)))
        except ValueError:
            year = month = day = -1
        if _valid_date(year, month, day):
            return year, month, day

    m = _TWO_DIGIT_DATE.search(text)
    if m:
        try:
            g1, g2, year = int(_fix(m.group(1))), int(_fix(m.group(2))), int(_fix(m.group(3)))
        except ValueError:
            return None
        resolved = _resolve_two_digit_date(g1, g2, year)
        if resolved:
            return resolved

    m = _COMPACT_DATE.search(text)
    if m:
        try:
            year, month, day = int(_fix(m.group(1))), int(_fix(m.group(2))), int(_fix(m.group(3)))
        except ValueError:
            return None
        if _valid_date(year, month, day):
            return year, month, day

    return None


def _find_time(text: str) -> tuple[int, int, int] | None:
    for pattern in (_TIME_COLON, _TIME_DASH):
        m = pattern.search(text)
        if not m:
            continue
        try:
            hour, minute, second = int(_fix(m.group(1))), int(_fix(m.group(2))), int(_fix(m.group(3)))
        except ValueError:
            continue
        ampm = (m.group(4) or "").upper() or None
        if ampm == "PM" and hour < 12:
            hour += 12
        elif ampm == "AM" and hour == 12:
            hour = 0
        if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
            return hour, minute, second
    return None


def parse_overlay_text(text: str) -> datetime | None:
    """Best-effort parse of a date+time burned into a CCTV frame.

    Tries ISO (``2026-07-23``), day-first and month-first two-digit-year-4
    layouts (``23/07/2026`` / ``07-23-2026``), and a compact ``20260723``
    date, each paired with a colon or dash time (optionally with AM/PM),
    tolerating common OCR digit misreads (``O``/``o`` -> ``0``, ``l``/``I``
    -> ``1``, etc). Returns ``None`` -- never raises -- when no plausible
    date+time combination is found, so callers can treat "OCR couldn't read
    this frame" as an ordinary, expected outcome.
    """
    if not text:
        return None

    date_ymd = _find_date(text)
    if date_ymd is None:
        return None

    time_hms = _find_time(text)
    if time_hms is None:
        return None

    year, month, day = date_ymd
    hour, minute, second = time_hms
    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


# --- imaging side (deferred easyocr/cv2 imports) -----------------------------

_reader_cache: dict[str, object] = {}


def get_reader(device: str | None = None):
    """Return a cached ``easyocr.Reader``, creating it once per device.

    Deferred import: importing ``easyocr`` (and the torch it pulls in) is
    slow, so this is only paid the first time OCR is actually requested.
    """
    key = device or "auto"
    reader = _reader_cache.get(key)
    if reader is None:
        import easyocr  # heavy import, deferred

        gpu = bool(device) and device != "cpu"
        reader = easyocr.Reader(list(config.OCR_LANGUAGES), gpu=gpu)
        _reader_cache[key] = reader
    return reader


def find_overlay_region(image, reader) -> tuple[int, int, int, int] | None:
    """OCR the whole frame once and return the bbox of the timestamp text.

    Returns ``(x1, y1, x2, y2)`` of the first text region that parses as a
    plausible date+time, or ``None`` if nothing in the frame does. Padded
    slightly so a tighter per-frame crop still fully contains the text.
    """
    h, w = image.shape[:2]
    for bbox, text, _conf in reader.readtext(image):
        if parse_overlay_text(text) is None:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        pad_x, pad_y = 10, 6
        x1 = max(0, int(min(xs)) - pad_x)
        y1 = max(0, int(min(ys)) - pad_y)
        x2 = min(w, int(max(xs)) + pad_x)
        y2 = min(h, int(max(ys)) + pad_y)
        return x1, y1, x2, y2
    return None


def read_timestamp(image, reader, region: tuple[int, int, int, int] | None = None) -> datetime | None:
    """OCR one frame (or just ``region`` of it) and parse a timestamp from it."""
    crop = image
    if region is not None:
        x1, y1, x2, y2 = region
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = image
    texts = reader.readtext(crop, detail=0)
    combined = " ".join(texts)
    ts = parse_overlay_text(combined)
    if ts is not None:
        return ts
    # Fall back to parsing each detected line individually -- some overlays
    # split date and time far enough apart that joining everything with a
    # single space produces a run the regexes don't line up on.
    for text in texts:
        ts = parse_overlay_text(text)
        if ts is not None:
            return ts
    return None


def sidecar_path(folder: str) -> str:
    return os.path.join(folder, config.TIMESTAMP_SIDECAR_NAME)


def load_sidecar(folder: str) -> dict[str, datetime]:
    """Read ``<folder>/<sidecar>`` if present; ``{}`` on missing/corrupt file."""
    path = sidecar_path(folder)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return {name: datetime.fromisoformat(iso) for name, iso in raw.items()}
    except Exception:
        log.warning("Timestamp sidecar %s unreadable, ignoring", path, exc_info=True)
        return {}


def _save_sidecar(folder: str, timestamps: dict[str, datetime]) -> None:
    path = sidecar_path(folder)
    raw = {name: ts.isoformat() for name, ts in timestamps.items()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(raw, fh, indent=2, sort_keys=True)


def ocr_folder(
    folder: str,
    filenames: list[str],
    device: str | None = None,
    max_consecutive_misses: int = 5,
    progress=None,
    reader=None,
) -> dict[str, datetime]:
    """OCR every frame in ``filenames`` and write the result as a sidecar.

    ``filenames`` are basenames within ``folder`` (already filtered to image
    files by the caller, e.g. ``frame_loader``). The overlay's on-screen
    location is located once from the first readable frame and reused for
    the rest (OCR-ing the full frame every time is far slower); if a run of
    ``max_consecutive_misses`` frames fails to parse from that region, the
    next frame falls back to a full-frame OCR pass in case the overlay moved.

    ``reader`` is normally left as ``None`` (a real ``easyocr.Reader`` is
    created via ``get_reader``); tests pass a fake with a ``readtext`` method
    to exercise this function without the ``easyocr`` dependency installed.

    Returns the ``{filename: datetime}`` mapping that was written to
    ``<folder>/<config.TIMESTAMP_SIDECAR_NAME>``; ``progress(done, total,
    message)`` is called once per frame processed for GUI/CLI feedback.
    """
    import cv2  # deferred so importing this module stays light

    if reader is None:
        reader = get_reader(device)
    region: tuple[int, int, int, int] | None = None
    results: dict[str, datetime] = {}
    misses_in_a_row = 0
    total = len(filenames)

    for idx, name in enumerate(filenames):
        path = os.path.join(folder, name)
        image = imread_unicode(cv2, path)
        if image is None:
            log.warning("OCR: could not read image %s (skipped)", path)
            if progress:
                progress(idx + 1, total, f"{name} (unreadable)")
            continue

        if region is None:
            region = find_overlay_region(image, reader)

        ts = read_timestamp(image, reader, region) if region is not None else None
        if ts is None and region is not None:
            # The known region stopped working (overlay moved, or this frame
            # is just noisy) -- retry once against the full frame.
            ts = read_timestamp(image, reader, region=None)
            if ts is not None:
                # Re-anchor to a fresh region so subsequent frames stay fast.
                region = find_overlay_region(image, reader) or region

        if ts is not None:
            results[name] = ts
            misses_in_a_row = 0
        else:
            misses_in_a_row += 1
            if misses_in_a_row >= max_consecutive_misses:
                # Repeated failures against the cached region -- drop it so
                # the next frame re-locates the overlay from scratch.
                region = None
                misses_in_a_row = 0

        if progress:
            progress(idx + 1, total, name)

    if results:
        _save_sidecar(folder, results)
    return results
