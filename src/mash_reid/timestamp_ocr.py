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
import dataclasses
from dataclasses import dataclass
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
# Some CCTV overlays render the time with dots -- "2026-07-24 17.40.50" -- same
# shape as _TIME_DASH, same negative lookbehind (so a dot-separated date like
# "2026.07.24" can't have its last two groups misread as a time).
_TIME_DOT = re.compile(rf"(?<!{_D})({_D}{{2}})\.({_D}{{2}})\.({_D}{{2}})\s*(AM|PM|am|pm)?")


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
    for pattern in (_TIME_COLON, _TIME_DASH, _TIME_DOT):
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


# --- user-configurable overlay pattern ---------------------------------------
#
# The built-in parser above tolerates the separator styles and OCR noise seen
# on common CCTV overlays, but some cameras use a layout it simply can't
# guess (a different field order, a written month, a 2-digit year meant
# unambiguously) -- and this is the escape hatch: a user-supplied template
# using a small token vocabulary rather than a raw regex, since the people
# hitting this are reading a Thai CCTV export, not writing Python.
#
# A pattern is written like the overlay itself, letters-for-digits:
#   "YYYY-MO-DD HH.MI.SS"   matches   "2026-07-24 17.40.50"
#   "DD/MO/YYYY hh:MI:SS AP" matches  "24/07/2026 05:40:50 PM"
# Every character that isn't a token is matched literally, so the user is
# really just describing their overlay's punctuation once.

_CUSTOM_TOKEN_WIDTH = {"YYYY": 4, "YY": 2, "MO": 2, "DD": 2,
                      "HH": 2, "hh": 2, "MI": 2, "SS": 2}
_CUSTOM_TOKEN_GROUP = {"YYYY": "year", "YY": "year2", "MO": "month", "DD": "day",
                       "HH": "hour24", "hh": "hour12", "MI": "minute", "SS": "second"}
# Longest first, so "YYYY" is matched before "YY" would greedily claim half of it.
_CUSTOM_TOKENS = sorted(_CUSTOM_TOKEN_WIDTH, key=len, reverse=True) + ["AP"]
_CUSTOM_TOKEN_SPLIT = re.compile("(" + "|".join(_CUSTOM_TOKENS) + ")")


def compile_custom_pattern(pattern: str) -> "re.Pattern":
    """Compile a user's token template into a regex with named groups.

    Raises ``ValueError`` with a message meant to be shown directly to the
    user when the template is unusable: empty, has no recognizable tokens,
    reuses a token, or is missing a field a date+time needs (year, month,
    day, one of an hour token, minute, second).
    """
    pattern = (pattern or "").strip()
    if not pattern:
        raise ValueError("Pattern is empty")

    parts: list[str] = []
    seen: set[str] = set()
    for chunk in _CUSTOM_TOKEN_SPLIT.split(pattern):
        if not chunk:
            continue
        if chunk == "AP":
            seen.add("ampm")
            parts.append(r"(?P<ampm>AM|PM|am|pm)")
        elif chunk in _CUSTOM_TOKEN_WIDTH:
            group = _CUSTOM_TOKEN_GROUP[chunk]
            if group in seen:
                raise ValueError(f"'{chunk}' appears more than once in the pattern")
            seen.add(group)
            parts.append(f"(?P<{group}>{_D}{{{_CUSTOM_TOKEN_WIDTH[chunk]}}})")
        else:
            parts.append(re.escape(chunk))

    if "year" not in seen and "year2" not in seen:
        raise ValueError("Pattern needs a year token (YYYY or YY)")
    if "month" not in seen or "day" not in seen:
        raise ValueError("Pattern needs MO (month) and DD (day) tokens")
    if "hour24" not in seen and "hour12" not in seen:
        raise ValueError("Pattern needs an hour token (HH for 24-hour, hh for 12-hour)")
    if "minute" not in seen or "second" not in seen:
        raise ValueError("Pattern needs MI (minute) and SS (second) tokens")
    if "hour12" in seen and "ampm" not in seen:
        raise ValueError("A 12-hour clock (hh) needs an AP (AM/PM) token too")

    return re.compile("".join(parts))


def _parse_with_custom_pattern(text: str, compiled: "re.Pattern") -> datetime | None:
    match = compiled.search(text)
    if match is None:
        return None
    groups = match.groupdict()

    try:
        if groups.get("year"):
            year = int(_fix(groups["year"]))
        else:
            year = 2000 + int(_fix(groups["year2"]))
        month = int(_fix(groups["month"]))
        day = int(_fix(groups["day"]))
        hour = int(_fix(groups["hour24"])) if groups.get("hour24") else int(_fix(groups["hour12"]))
        minute = int(_fix(groups["minute"]))
        second = int(_fix(groups["second"]))
    except (ValueError, TypeError):
        return None

    ampm = (groups.get("ampm") or "").upper() or None
    if ampm == "PM" and hour < 12:
        hour += 12
    elif ampm == "AM" and hour == 12:
        hour = 0

    if not _valid_date(year, month, day) or not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return None
    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


def default_patterns_path() -> str:
    """``<project root>/ocr_patterns.json`` -- next to ``config.py``."""
    root = os.path.dirname(os.path.abspath(config.__file__))
    return os.path.join(root, config.DEFAULT_OCR_PATTERNS_FILE)


def load_saved_patterns(path: str | None = None) -> dict[str, str]:
    """User-saved named overlay patterns, on top of ``config.BUILTIN_OCR_TIME_PATTERNS``.

    Best-effort, matching ``mash_reid.settings``'s discipline: a missing or
    corrupt file just means "nothing saved yet", never a crash. Values that
    aren't plain strings (a hand-edited or foreign file) are dropped rather
    than passed through to a later ``compile_custom_pattern`` call.
    """
    path = path or default_patterns_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception:
        log.warning("Saved OCR patterns file %s unreadable, ignoring", path, exc_info=True)
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def save_patterns(patterns: dict[str, str], path: str | None = None) -> None:
    """Write ``patterns`` (name -> token template) to ``path`` as JSON.

    Never raises -- a failed save is logged and otherwise silently skipped,
    same reasoning as ``mash_reid.settings.save``: losing a saved pattern is
    far less costly than crashing the dialog that manages them.
    """
    path = path or default_patterns_path()
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(patterns, fh, indent=2, sort_keys=True, ensure_ascii=False)
    except Exception:
        log.warning("Could not write OCR patterns to %s", path, exc_info=True)


def strip_pattern_label(text: str) -> str:
    """"YYYY-MO-DD HH:MI:SS   (ISO date, colon time)" -> "YYYY-MO-DD HH:MI:SS".

    The pattern picker's dropdown shows entries as "pattern   (label)" so a
    choice is both immediately usable and identifiable; this recovers the
    bare, compilable pattern regardless of whether the text came from
    picking one of those entries or was typed freehand (which round-trips
    unchanged, since it has no trailing annotation to strip).
    """
    text = text.strip()
    if text.endswith(")") and "   (" in text:
        text = text.rsplit("   (", 1)[0].strip()
    return text


def pattern_choices() -> list[str]:
    """Combobox values for the pattern picker: built-ins, then saved ones."""
    choices = [f"{pattern}   ({label})" for label, pattern in config.BUILTIN_OCR_TIME_PATTERNS]
    choices += [f"{pattern}   ({name})" for name, pattern in sorted(load_saved_patterns().items())]
    return choices


def parse_overlay_text(text: str, custom_pattern: "re.Pattern | None" = None) -> datetime | None:
    """Best-effort parse of a date+time burned into a CCTV frame.

    Tries ISO (``2026-07-23``), day-first and month-first two-digit-year-4
    layouts (``23/07/2026`` / ``07-23-2026``), and a compact ``20260723``
    date, each paired with a colon, dash or dot time (optionally with AM/PM),
    tolerating common OCR digit misreads (``O``/``o`` -> ``0``, ``l``/``I``
    -> ``1``, etc). Returns ``None`` -- never raises -- when no plausible
    date+time combination is found, so callers can treat "OCR couldn't read
    this frame" as an ordinary, expected outcome.

    ``custom_pattern`` -- compiled via ``compile_custom_pattern`` -- is tried
    *first* when given: it represents a format the user has told us their
    camera actually uses, so it's authoritative over the generic guesses
    above. Falls through to the built-in patterns if it doesn't match this
    particular string (OCR text varies frame to frame even for one camera).
    """
    if not text:
        return None

    if custom_pattern is not None:
        ts = _parse_with_custom_pattern(text, custom_pattern)
        if ts is not None:
            return ts

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


def _reader_gpu_flag(device: str | None = None,
                     cuda_available: bool | None = None) -> tuple[str, bool]:
    """Resolve ``device`` and return ``(resolved, gpu_flag_for_easyocr)``.

    Split out from ``get_reader`` so it can be tested without easyocr (or a
    GPU) installed -- ``cuda_available`` is forwarded to
    ``device.resolve_device``, the same injection point the rest of the
    package uses.

    This used to be ``gpu = bool(device) and device != "cpu"``, which got the
    answer wrong in both directions: the GUI passes ``"auto"``, which is
    truthy, so easyocr was told to use a GPU on machines that have none,
    while the CLI's default of ``None`` is falsy, so it was told *not* to use
    a GPU on machines that do. Resolving first fixes both, since "auto" has
    already collapsed to a concrete "cpu"/"cuda" by the time it's inspected.
    """
    from mash_reid.device import resolve_device  # local: keeps import graph flat

    resolved = resolve_device(device, cuda_available=cuda_available)
    return resolved, resolved != "cpu"


def get_reader(device: str | None = None):
    """Return a cached ``easyocr.Reader``, creating it once per device.

    Deferred import: importing ``easyocr`` (and the torch it pulls in) is
    slow, so this is only paid the first time OCR is actually requested.

    The cache is keyed by the *resolved* device rather than the requested
    one, so "auto" and "cuda" share a single reader instead of loading two
    copies of the OCR model into the same VRAM.
    """
    resolved, gpu = _reader_gpu_flag(device)
    reader = _reader_cache.get(resolved)
    if reader is None:
        import easyocr  # heavy import, deferred

        reader = easyocr.Reader(list(config.OCR_LANGUAGES), gpu=gpu)
        _reader_cache[resolved] = reader
    return reader


def find_overlay_region(image, reader, custom_pattern=None) -> tuple[int, int, int, int] | None:
    """OCR the whole frame once and return the bbox of the timestamp text.

    Returns ``(x1, y1, x2, y2)`` of the *most confident* text region that
    parses as a plausible date+time, or ``None`` if nothing in the frame
    does. Padded slightly so a tighter per-frame crop still fully contains
    the text.
    """
    h, w = image.shape[:2]
    best = None
    for bbox, text, conf in reader.readtext(image):
        if parse_overlay_text(text, custom_pattern) is None:
            continue
        if best is None or conf > best[0]:
            best = (conf, bbox)
    if best is None:
        return None

    xs = [p[0] for p in best[1]]
    ys = [p[1] for p in best[1]]
    pad_x, pad_y = config.OCR_REGION_PAD
    return (max(0, int(min(xs)) - pad_x), max(0, int(min(ys)) - pad_y),
            min(w, int(max(xs)) + pad_x), min(h, int(max(ys)) + pad_y))


@dataclass(frozen=True)
class TimestampCandidate:
    """One possible reading of a frame's clock, with how sure the OCR was.

    Unparseable readings are kept deliberately: when a user is deciding
    whether a frame is legible at all, seeing the raw ``"2O26-O7-Z3 lO:l5"``
    the OCR produced is far more informative than an empty list.
    """

    text: str
    timestamp: datetime | None
    confidence: float
    origin: str  # "joined" | "line"


def _crop_to_region(image, region):
    if region is None:
        return image
    x1, y1, x2, y2 = region
    crop = image[y1:y2, x1:x2]
    return image if crop.size == 0 else crop


def read_timestamp_candidates(
    image, reader, region: tuple[int, int, int, int] | None = None, custom_pattern=None,
) -> list[TimestampCandidate]:
    """Every plausible reading of one frame's clock, best first.

    A single ``readtext`` call (``detail=1``, so the per-detection confidence
    easyocr already computes is kept rather than thrown away) produces both
    the joined-lines candidate and the per-line ones. Ranking is by "does it
    parse" first, then confidence, and is *stable*, so the joined reading
    stays ahead of an equally-confident single line -- preserving the
    original resolution order.

    The joined candidate's confidence is the **minimum** across its lines,
    not the mean: one badly-read glyph anywhere in the joined string can flip
    a digit, so the weakest link is the honest score.

    ``custom_pattern`` (from ``compile_custom_pattern``) is passed straight
    through to ``parse_overlay_text`` for every candidate.
    """
    crop = _crop_to_region(image, region)
    detections = reader.readtext(crop)

    texts, confs = [], []
    for _bbox, text, conf in detections:
        texts.append(text)
        confs.append(float(conf))

    candidates: list[TimestampCandidate] = []
    if texts:
        joined = " ".join(texts)
        candidates.append(TimestampCandidate(
            text=joined, timestamp=parse_overlay_text(joined, custom_pattern),
            confidence=min(confs), origin="joined"))
    for text, conf in zip(texts, confs):
        candidates.append(TimestampCandidate(
            text=text, timestamp=parse_overlay_text(text, custom_pattern),
            confidence=conf, origin="line"))

    candidates.sort(key=lambda c: (c.timestamp is not None, c.confidence), reverse=True)
    return candidates


def reparse_candidates(candidates: list[TimestampCandidate],
                       custom_pattern=None) -> list[TimestampCandidate]:
    """Re-run parsing against already-OCR'd text, e.g. after the user adds a
    custom pattern -- no new OCR call, since the raw text is already in hand.

    Re-ranked the same way ``read_timestamp_candidates`` ranks a fresh read,
    so a pattern that newly makes the joined text parseable moves it back to
    the front.
    """
    reparsed = [dataclasses.replace(c, timestamp=parse_overlay_text(c.text, custom_pattern))
               for c in candidates]
    reparsed.sort(key=lambda c: (c.timestamp is not None, c.confidence), reverse=True)
    return reparsed


def read_timestamp(image, reader, region: tuple[int, int, int, int] | None = None,
                   custom_pattern=None) -> datetime | None:
    """OCR one frame (or just ``region`` of it) and parse a timestamp from it.

    Thin wrapper over ``read_timestamp_candidates`` -- kept because the whole
    per-frame path (``ocr_folder``) still uses it, and its behaviour is
    unchanged: the joined text is tried first, then each detected line, since
    some overlays split date and time far enough apart that joining them
    produces a run the regexes don't line up on.
    """
    for candidate in read_timestamp_candidates(image, reader, region, custom_pattern):
        if candidate.timestamp is not None:
            return candidate.timestamp
    return None


def sidecar_path(folder: str) -> str:
    return os.path.join(folder, config.TIMESTAMP_SIDECAR_NAME)


def _parse_iso_map(raw: dict) -> dict[str, datetime]:
    """Parse a ``{name: iso}`` map, skipping individual bad entries.

    Deliberately per-entry: the previous version wrapped the whole
    comprehension in one ``try``, so a single malformed value threw away all
    12,000 timestamps and silently fell the entire folder back to filename
    times. One bad entry should cost one frame, not the folder.
    """
    out: dict[str, datetime] = {}
    bad = 0
    for name, iso in raw.items():
        try:
            out[name] = datetime.fromisoformat(iso)
        except (TypeError, ValueError):
            bad += 1
    if bad:
        log.warning("Timestamp sidecar: skipped %d unparseable timestamp(s)", bad)
    return out


def load_sidecar_doc(folder: str) -> dict | None:
    """The raw sidecar JSON, or ``None`` when missing/unreadable.

    Only file-level failures (missing, unreadable, not JSON, not an object)
    collapse to ``None`` -- malformed *entries* are handled further in, so a
    partially-damaged file still yields whatever is intact.
    """
    path = sidecar_path(folder)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except Exception:
        log.warning("Timestamp sidecar %s unreadable, ignoring", path, exc_info=True)
        return None
    if not isinstance(raw, dict):
        log.warning("Timestamp sidecar %s is not a JSON object, ignoring", path)
        return None
    return raw


def _frames_map(raw: dict) -> dict:
    """The name->iso mapping, whichever schema version this file uses.

    v2 nests it under ``frames`` alongside the fit metadata; v1 *is* the
    mapping. A v1 file can't be mistaken for v2 because its values are ISO
    strings rather than a nested object, and vice versa.
    """
    if isinstance(raw.get("version"), int):
        frames = raw.get("frames")
        return frames if isinstance(frames, dict) else {}
    return raw


def load_sidecar(folder: str) -> dict[str, datetime]:
    """Read ``<folder>/<sidecar>`` if present; ``{}`` on missing/corrupt file.

    Reads both schema versions, so sidecars written by earlier versions keep
    working with no migration step.
    """
    raw = load_sidecar_doc(folder)
    if raw is None:
        return {}
    return _parse_iso_map(_frames_map(raw))


def load_sidecar_source(folder: str) -> str:
    """Where a folder's sidecar timestamps came from: ``"timeline"`` or ``"ocr"``.

    Surfaced per-frame as ``Frame.timestamp_source`` so the GUI can show at a
    glance whether times came from a reviewed fit or from the older
    every-frame OCR pass.
    """
    raw = load_sidecar_doc(folder)
    if raw is None:
        return "ocr"
    source = raw.get("source")
    return source if isinstance(source, str) and source else "ocr"


def write_sidecar_doc(folder: str, doc: dict) -> None:
    """Write a v2 sidecar atomically.

    Via a temp file + ``os.replace`` so an interrupted write can't leave a
    truncated sidecar behind -- unlike the per-frame path, this file is now
    the *only* source of times for every frame in the folder, so a partial
    one would silently downgrade the whole folder to filename times.
    """
    path = sidecar_path(folder)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


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
    custom_pattern=None,
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

    ``custom_pattern`` (from ``compile_custom_pattern``) is forwarded to
    every parse, for overlay formats the built-in parser can't guess.

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
            region = find_overlay_region(image, reader, custom_pattern)

        ts = (read_timestamp(image, reader, region, custom_pattern)
              if region is not None else None)
        if ts is None and region is not None:
            # The known region stopped working (overlay moved, or this frame
            # is just noisy) -- retry once against the full frame.
            ts = read_timestamp(image, reader, region=None, custom_pattern=custom_pattern)
            if ts is not None:
                # Re-anchor to a fresh region so subsequent frames stay fast.
                region = find_overlay_region(image, reader, custom_pattern) or region

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


# --- sampled scan (the fast path) --------------------------------------------

# How many frames to try before giving up on locating the overlay. Probing
# only the first frame is fragile -- it may be dark, or a vehicle may be
# sitting over the clock -- and a full-frame readtext is ~10x a cropped one,
# so this is deliberately a small number spread across the set.
REGION_PROBE_LIMIT = 5


@dataclass
class ScanResult:
    """Everything the review dialog needs, from one pass over the samples."""

    folder: str
    samples: list          # list[timeline.TimeSample]
    keys: list             # list[timeline.FrameKey] -- every frame in the folder
    x_kind: str
    fit: object            # timeline.TimelineFit
    crops: dict            # name -> the clock-region image, for display
    unparsed: list         # filenames excluded from the axis
    region: tuple | None
    region_is_manual: bool = False


def _probe_indices(count: int, limit: int = REGION_PROBE_LIMIT) -> list[int]:
    """Spread-out sample positions to try when locating the overlay."""
    if count <= 0:
        return []
    if count <= limit:
        return list(range(count))
    return sorted({0, count - 1, count // 2, count // 4, 3 * count // 4})


def scan_folder(
    folder: str,
    filenames: list[str],
    device: str | None = None,
    sample_count: int = config.DEFAULT_TIMELINE_SAMPLE_TARGET,
    min_per_clip: int = config.DEFAULT_TIMELINE_MIN_SAMPLES_PER_CLIP,
    progress=None,
    reader=None,
    custom_pattern=None,
    region_override: tuple[int, int, int, int] | None = None,
) -> ScanResult:
    """OCR a sample of ``filenames`` and fit a timeline through the readings.

    The fast replacement for ``ocr_folder``: instead of reading every frame's
    clock, read ~``sample_count`` of them and solve for the line they lie on
    (see ``mash_reid.timeline``). Two to three orders of magnitude less OCR,
    and more accurate, since misreads land off the line and are discarded
    instead of committed.

    ``custom_pattern`` (from ``compile_custom_pattern``) is forwarded to every
    parse. ``region_override`` -- normally the box a user drew by hand around
    the clock -- skips the automatic region probe entirely: it is both faster
    (no full-frame OCR pass to locate the overlay) and more reliable, since
    auto-detection can lock onto the wrong text (a camera-model watermark, a
    license plate) when nothing in the frame parses as a date+time yet.

    Writes nothing -- the caller reviews the fit and decides. ``progress(done,
    total, message)`` matches the callback shape used everywhere else here.
    """
    import cv2  # deferred so importing this module stays light

    from mash_reid import timeline

    if reader is None:
        reader = get_reader(device)

    keys, x_kind, unparsed = timeline.build_keys(filenames)
    picked = timeline.select_samples(keys, total_target=sample_count,
                                     min_per_clip=min_per_clip)
    total = len(picked)

    region = region_override
    images: dict[str, object] = {}
    if region is None:
        # Locate the overlay once, from whichever probe frame yields a
        # parseable clock, then reuse that region for every sample.
        for probe in _probe_indices(total):
            name = picked[probe].name
            image = images.get(name)
            if image is None:
                image = imread_unicode(cv2, os.path.join(folder, name))
                if image is None:
                    continue
                images[name] = image
            region = find_overlay_region(image, reader, custom_pattern)
            if region is not None:
                break

    samples, crops = [], {}
    for idx, key in enumerate(picked):
        image = images.pop(key.name, None)
        if image is None:
            image = imread_unicode(cv2, os.path.join(folder, key.name))
        if image is None:
            log.warning("Scan: could not read image %s (skipped)", key.name)
            if progress:
                progress(idx + 1, total, f"{key.name} (unreadable)")
            continue

        candidates = read_timestamp_candidates(image, reader, region, custom_pattern)
        chosen = next((c.timestamp for c in candidates if c.timestamp), None)
        samples.append(timeline.TimeSample(
            name=key.name, clip=key.clip, x=key.x, chosen=chosen,
            filename_time=key.filename_time, candidates=tuple(candidates)))
        crops[key.name] = _crop_to_region(image, region).copy()

        if progress:
            progress(idx + 1, total, key.name)

    fit = timeline.fit_timeline(samples, x_kind=x_kind, all_keys=keys)
    return ScanResult(folder=folder, samples=samples, keys=keys, x_kind=x_kind,
                      fit=fit, crops=crops, unparsed=unparsed, region=region,
                      region_is_manual=region_override is not None)


# --- reopening a saved review, without OCR -----------------------------------

def _parse_iso_or_none(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def load_scan_for_review(folder: str, device: str | None = None,
                         reader=None) -> "ScanResult | None":
    """Rebuild a ``ScanResult`` from a saved v2 sidecar -- no OCR call.

    Lets "Fix times..." reopen exactly where a previous review left off
    (including any manual corrections and the region that was used) so the
    user isn't stuck re-reading the clock every time they want to keep
    adjusting a folder's times. The only work here is re-reading each
    sampled frame off disk and re-cropping it for display -- the OCR
    readings themselves came from the sidecar, not a fresh ``readtext`` call.

    Returns ``None`` when there's nothing to reopen: no sidecar, a v1
    (pre-fit) sidecar, or a v2 sidecar with no ``samples`` recorded (e.g. one
    written by the plain ``ocr_folder`` path some other way).
    """
    from mash_reid import timeline

    raw = load_sidecar_doc(folder)
    if raw is None or not isinstance(raw.get("version"), int):
        return None
    raw_samples = raw.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        return None

    import cv2  # deferred, and only needed once we know there's something to crop

    names = sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in config.IMAGE_EXTENSIONS
    )
    keys, x_kind, unparsed = timeline.build_keys(names)

    region = raw.get("region")
    region = tuple(region) if isinstance(region, list) and len(region) == 4 else None

    samples: list[timeline.TimeSample] = []
    crops: dict[str, object] = {}
    for entry in raw_samples:
        name = entry.get("name")
        if not name:
            continue
        candidates = tuple(
            TimestampCandidate(
                text=c.get("text", ""), timestamp=_parse_iso_or_none(c.get("time")),
                confidence=float(c.get("conf", 0.0)), origin="line")
            for c in entry.get("candidates", []) if isinstance(c, dict)
        )
        samples.append(timeline.TimeSample(
            name=name, clip=int(entry.get("clip", 0)), x=int(entry.get("x", 0)),
            chosen=_parse_iso_or_none(entry.get("chosen_time")),
            filename_time=_parse_iso_or_none(entry.get("filename_time")),
            candidates=candidates,
            edited=bool(entry.get("edited", False)),
            ignored=bool(entry.get("ignored", False)),
        ))

        image = imread_unicode(cv2, os.path.join(folder, name))
        if image is not None:
            crops[name] = _crop_to_region(image, region).copy()

    fit = timeline.fit_timeline(samples, x_kind=x_kind, all_keys=keys)
    return ScanResult(folder=folder, samples=samples, keys=keys, x_kind=x_kind,
                      fit=fit, crops=crops, unparsed=unparsed, region=region,
                      region_is_manual=bool(raw.get("region_is_manual", False)))
