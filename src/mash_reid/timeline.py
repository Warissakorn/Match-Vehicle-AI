"""Recover frame timestamps by fitting a line, instead of OCR-ing every frame.

The problem this replaces: ``timestamp_ocr.ocr_folder`` reads the burned-in
clock on *every* frame and commits each reading independently. On a real
gallery that is 12,000+ OCR calls (tens of minutes), and because there is no
cross-frame check, a single misread digit is accepted silently. Worse, frames
whose OCR fails keep their *filename* time -- a different clock -- so one
folder ends up carrying two time bases, and since ``frame_loader.load_frames``
sorts by timestamp, a bad reading reorders the sequence rather than just
skewing it.

The observation that fixes all of it: frames extracted by
``video_extractor`` are sampled at a fixed interval and carry their **source
frame index** in the filename, and a CCTV clock advances linearly with that
index::

    true_time = clip_start + frame_index / fps

That is a straight line with two unknowns per clip. Reading 12,000 frames to
learn two numbers is the actual waste. Sampling ~40 frames and fitting a
robust line is roughly 300x less OCR work *and* more accurate, because a
misread lands far from the line and is discarded by construction rather than
committed. Frames the OCR could not read at all still get a correct time from
the fit, which is what eliminates the mixed-time-base bug.

Everything here is pure: stdlib plus numpy, no cv2/easyocr/torch, no file
I/O. The OCR side lives in ``timestamp_ocr``; this module only ever sees
``(clip, x, datetime)`` triples, which is what makes the estimator fully
testable without images or a GPU.

**This is not a substitute for human confirmation.** A robust fit defends
against *independent* misreads. It cannot detect a *systematic* one -- if the
overlay font makes every ``8`` read as ``6``, the readings stay mutually
consistent and the fit looks excellent while being uniformly wrong. That is
why the review dialog shows the filename time beside the read time: the
difference between the two clocks is the thing a human can actually judge.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Sequence

import numpy as np

import config

log = logging.getLogger(__name__)

# Names produced by ``video_extractor.frame_filename``:
#   A_20260723_101530_000123.jpg          (single clip)
#   A_20260723_101530_c01_000123.jpg      (multi-clip)
# The point prefix is non-greedy because a point name may itself contain "_";
# the fixed-width date/time block anchors it. Clip and index widths are ">="
# their format widths, since %02d/%06d are minimums, not maximums.
_FRAME_NAME = re.compile(
    r"^(?P<point>.+?)_(?P<date>\d{8})_(?P<time>\d{6})"
    r"(?:_c(?P<clip>\d{2,}))?_(?P<idx>\d{6,})\.(?P<ext>[A-Za-z0-9]+)$"
)

X_FRAME_INDEX = "frame_index"
X_ORDINAL = "ordinal"

# Fraction of a folder's names that must parse before we trust frame indices.
_FRAME_INDEX_COVERAGE = 0.95


@dataclass(frozen=True)
class FrameKey:
    """One frame's position on the fit's x-axis."""

    name: str
    clip: int
    x: int
    filename_time: datetime | None = None


@dataclass(frozen=True)
class TimeSample:
    """A frame we OCR'd, and the reading currently chosen for it.

    ``candidates`` is opaque here -- the fit only looks at ``chosen``. It is
    carried through so the review dialog can offer the alternatives without a
    second pass over the images.
    """

    name: str
    clip: int
    x: int
    chosen: datetime | None
    filename_time: datetime | None = None
    candidates: tuple = ()
    edited: bool = False
    ignored: bool = False

    @property
    def usable(self) -> bool:
        return self.chosen is not None and not self.ignored


@dataclass(frozen=True)
class ClipFit:
    """Where one clip's timeline sits, given the shared slope."""

    clip: int
    intercept_seconds: float  # y at x=0, relative to TimelineFit.epoch
    start_time: datetime
    n_samples: int
    n_inliers: int
    residual_mad: float
    max_abs_residual: float
    user_offset_seconds: float = 0.0
    fitted: bool = True  # False -> intercept borrowed from the filename clock


@dataclass(frozen=True)
class TimelineFit:
    slope: float  # seconds per x unit
    x_kind: str
    epoch: datetime
    clips: dict[int, ClipFit]
    n_samples: int
    n_inliers: int
    residual_mad: float
    max_abs_residual: float
    implied_fps: float | None
    ok: bool
    warnings: tuple[str, ...] = ()
    residuals: dict[str, float] = field(default_factory=dict)
    inliers: frozenset[str] = frozenset()

    @property
    def inlier_fraction(self) -> float:
        return self.n_inliers / self.n_samples if self.n_samples else 0.0


# --- filename -> x -----------------------------------------------------------

def parse_frame_key(name: str) -> FrameKey | None:
    """Pull ``(clip, frame_index)`` out of an extractor-produced filename.

    Returns ``None`` for anything that doesn't match -- frames exported by
    other tools, or hand-renamed files. Callers fall back to ordinal
    positions for those (see ``build_keys``).
    """
    match = _FRAME_NAME.match(name)
    if match is None:
        return None
    try:
        filename_time = datetime.strptime(
            f"{match['date']}_{match['time']}", "%Y%m%d_%H%M%S")
    except ValueError:
        # Matches the shape but isn't a real date (e.g. month 13).
        filename_time = None
    clip = int(match["clip"]) if match["clip"] is not None else 0
    return FrameKey(name=name, clip=clip, x=int(match["idx"]),
                    filename_time=filename_time)


def build_keys(names: Sequence[str]) -> tuple[list[FrameKey], str, list[str]]:
    """Map filenames onto the fit's x-axis.

    Returns ``(keys, x_kind, unparsed_names)``.

    When nearly every name carries a frame index we use it, and the few that
    don't are reported so the caller can tell the user they're being left out
    -- they are genuinely not from this extraction, and mixing them in would
    corrupt the axis. Otherwise (frames from some other tool) we fall back to
    each file's ordinal position in sorted order, which is equivalent as long
    as the spacing is uniform. The two kinds are never mixed within a folder:
    their slopes are in different units (seconds per *source* frame vs
    seconds per *extracted* frame).
    """
    ordered = sorted(names)
    parsed = [parse_frame_key(n) for n in ordered]
    good = [k for k in parsed if k is not None]
    unparsed = [n for n, k in zip(ordered, parsed) if k is None]

    if ordered and len(good) >= 2 and len(good) >= _FRAME_INDEX_COVERAGE * len(ordered):
        return good, X_FRAME_INDEX, unparsed

    keys = [FrameKey(name=n, clip=0, x=i,
                     filename_time=(parsed[i].filename_time if parsed[i] else None))
            for i, n in enumerate(ordered)]
    return keys, X_ORDINAL, list(ordered) if not good else unparsed


# --- sampling ----------------------------------------------------------------

def select_samples(
    keys: Sequence[FrameKey],
    total_target: int = config.DEFAULT_TIMELINE_SAMPLE_TARGET,
    min_per_clip: int = config.DEFAULT_TIMELINE_MIN_SAMPLES_PER_CLIP,
) -> list[FrameKey]:
    """Pick the frames to OCR: evenly spread, per clip, endpoints included.

    ``total_target`` is a floor rather than a cap -- spreading 40 samples
    across a dozen clips would leave 3 each, too few to place a clip's start
    time. Each clip gets at least ``min_per_clip`` (or all of its frames, if
    it has fewer).

    Endpoints matter disproportionately: the slope's variance falls with the
    square of the x-span, so the first and last frame of a clip carry most of
    the information. ``linspace`` includes both.

    Fully deterministic, so reopening a review reproduces the same samples.
    """
    by_clip: dict[int, list[FrameKey]] = {}
    for key in keys:
        by_clip.setdefault(key.clip, []).append(key)
    if not by_clip:
        return []

    total = len(keys)
    chosen: list[FrameKey] = []
    for clip in sorted(by_clip):
        members = sorted(by_clip[clip], key=lambda k: k.x)
        share = round(total_target * len(members) / total) if total else 0
        want = max(share, min_per_clip)
        want = min(want, len(members))
        if want <= 0:
            continue
        if want == 1:
            picks = [0]
        else:
            picks = np.unique(
                np.linspace(0, len(members) - 1, want).round().astype(int))
        chosen.extend(members[i] for i in picks)
    return chosen


# --- the robust fit ----------------------------------------------------------

def _pairwise_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Slopes of every pair of points, skipping pairs that share an x."""
    if len(x) < 2:
        return np.empty(0, dtype=np.float64)
    i, j = np.triu_indices(len(x), k=1)
    dx = x[j] - x[i]
    ok = dx != 0
    if not ok.any():
        return np.empty(0, dtype=np.float64)
    return (y[j] - y[i])[ok] / dx[ok]


def theil_sen(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Median-of-pairwise-slopes line fit; returns ``(slope, intercept)``.

    Chosen over least squares because one OCR hour-digit misread is a 3600 s
    outlier, and least squares spreads that error across the whole line --
    a single bad reading at one end of a multi-hour clip skews the slope
    enough to drift the far end by tens of seconds.

    Chosen over RANSAC because at n=40 there are only 780 two-point
    hypotheses, so exhaustive RANSAC *is* this, plus an inlier threshold, an
    iteration budget and a random seed to tune. Theil-Sen is deterministic
    and parameter-free, which matters when a human is being asked to confirm
    the number it produces.

    Breakdown point is 29.3%, i.e. it tolerates ~11 bad readings out of 40.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    slopes = _pairwise_slopes(xa, ya)
    if slopes.size == 0:
        return 0.0, float(np.median(ya)) if ya.size else 0.0
    slope = float(np.median(slopes))
    return slope, float(np.median(ya - slope * xa))


def _pooled_slope(groups: list[tuple[np.ndarray, np.ndarray]]) -> float | None:
    """One slope shared by every clip: median of all *within-clip* pair slopes.

    Clips share a camera and encoder, so they share ``1/fps``; only the start
    time differs. Sharing the slope is not just tidy, it is what makes the
    fit work at all -- 40 samples over a dozen clips is ~3 each, far too few
    for a per-clip slope, while a per-clip *intercept* from 3 points is fine.

    Pairs are never formed across clips: ``x`` restarts at each clip, so a
    cross-clip pair's slope is meaningless. Pooling the pair slopes (rather
    than averaging per-clip medians) weights each clip by how much evidence
    it actually contributes.
    """
    parts = [s for x, y in groups if (s := _pairwise_slopes(x, y)).size]
    if not parts:
        return None
    return float(np.median(np.concatenate(parts)))


def _clip_intercepts(groups: dict[int, tuple[np.ndarray, np.ndarray]],
                     slope: float) -> dict[int, float]:
    return {clip: float(np.median(y - slope * x)) for clip, (x, y) in groups.items()}


def fit_timeline(
    samples: Sequence[TimeSample],
    x_kind: str = X_FRAME_INDEX,
    all_keys: Sequence[FrameKey] | None = None,
    sigma: float = config.TIMELINE_INLIER_SIGMA,
    tol_floor: float = config.TIMELINE_INLIER_FLOOR_SECONDS,
) -> TimelineFit:
    """Fit one shared slope plus a per-clip intercept over the OCR readings.

    ``all_keys`` (every frame in the folder, not just the sampled ones) lets
    clips that produced no readable sample still be placed -- see
    ``_borrow_intercepts``. Without it those clips are simply absent from the
    result and their frames keep whatever time they already had.

    Never raises: degenerate input (no samples, one sample, all-identical x)
    comes back as ``ok=False`` with an explanatory warning, because this is
    fed straight into a UI.
    """
    usable = [s for s in samples if s.usable]
    if not usable:
        return _failed_fit(x_kind, len(samples),
                           ("No frame produced a readable timestamp",))

    epoch = _epoch_for(usable)
    groups = _group(usable, epoch)

    slope = _pooled_slope(list(groups.values()))
    if slope is None:
        return _failed_fit(
            x_kind, len(usable),
            ("Not enough distinct frames to establish a rate "
             "(every reading came from the same position)",), epoch=epoch)

    intercepts = _clip_intercepts(groups, slope)
    residuals = _residuals(usable, epoch, slope, intercepts)

    # Scale estimate for outlier rejection. The floor is load-bearing: the
    # overlay is quantized to whole seconds, so on clean footage the MAD is
    # exactly 0, the tolerance would collapse to 0, and every sample that
    # wasn't bit-exact would be rejected -- leaving the refit with one or two
    # points and producing nonsense.
    mad = float(np.median(np.abs(residuals))) if residuals.size else 0.0
    tol = max(sigma * 1.4826 * mad, tol_floor)
    inlier_mask = np.abs(residuals) <= tol

    # Refit on inliers only, then score every sample against the new line.
    inliers = [s for s, keep in zip(usable, inlier_mask) if keep]
    if len(inliers) >= 2:
        in_groups = _group(inliers, epoch)
        refit_slope = _pooled_slope(list(in_groups.values()))
        if refit_slope is not None:
            slope = refit_slope
        intercepts = _clip_intercepts(in_groups, slope)
        # Clips that lost every sample to the filter still need placing;
        # fall back to all of their readings rather than dropping the clip.
        for clip, (x, y) in groups.items():
            intercepts.setdefault(clip, float(np.median(y - slope * x)))
        residuals = _residuals(usable, epoch, slope, intercepts)
        mad = float(np.median(np.abs(residuals))) if residuals.size else 0.0
        tol = max(sigma * 1.4826 * mad, tol_floor)
        inlier_mask = np.abs(residuals) <= tol

    inlier_names = frozenset(s.name for s, keep in zip(usable, inlier_mask) if keep)
    clips = _build_clip_fits(usable, epoch, intercepts, residuals, inlier_names)
    if all_keys:
        clips = _borrow_intercepts(clips, all_keys, epoch, slope)

    max_abs = float(np.max(np.abs(residuals))) if residuals.size else 0.0
    implied_fps = (1.0 / slope) if (x_kind == X_FRAME_INDEX and slope > 0) else None

    fit = TimelineFit(
        slope=slope, x_kind=x_kind, epoch=epoch, clips=clips,
        n_samples=len(usable), n_inliers=int(inlier_mask.sum()),
        residual_mad=mad, max_abs_residual=max_abs, implied_fps=implied_fps,
        ok=slope > 0,
        residuals={s.name: float(r) for s, r in zip(usable, residuals)},
        inliers=inlier_names,
    )
    return _with_warnings(fit, x_kind)


def _failed_fit(x_kind: str, n_samples: int, warnings: tuple[str, ...],
                epoch: datetime | None = None) -> TimelineFit:
    return TimelineFit(
        slope=0.0, x_kind=x_kind, epoch=epoch or datetime(1970, 1, 1), clips={},
        n_samples=n_samples, n_inliers=0, residual_mad=0.0, max_abs_residual=0.0,
        implied_fps=None, ok=False, warnings=warnings)


def _epoch_for(samples: Sequence[TimeSample]) -> datetime:
    """A nearby origin, so the fit works in small float64 second offsets."""
    times = sorted(s.chosen for s in samples)
    mid = times[len(times) // 2]
    return mid.replace(second=0, microsecond=0)


def _group(samples: Sequence[TimeSample],
           epoch: datetime) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per-clip ``(x, seconds-since-epoch)`` arrays."""
    buckets: dict[int, list[tuple[float, float]]] = {}
    for s in samples:
        buckets.setdefault(s.clip, []).append(
            (float(s.x), (s.chosen - epoch).total_seconds()))
    return {clip: (np.array([p[0] for p in pts], dtype=np.float64),
                   np.array([p[1] for p in pts], dtype=np.float64))
            for clip, pts in buckets.items()}


def _residuals(samples: Sequence[TimeSample], epoch: datetime, slope: float,
               intercepts: dict[int, float]) -> np.ndarray:
    return np.array(
        [(s.chosen - epoch).total_seconds() - (slope * s.x + intercepts.get(s.clip, 0.0))
         for s in samples], dtype=np.float64)


def _build_clip_fits(samples, epoch, intercepts, residuals, inlier_names) -> dict[int, ClipFit]:
    per_clip: dict[int, list[tuple[TimeSample, float]]] = {}
    for s, r in zip(samples, residuals):
        per_clip.setdefault(s.clip, []).append((s, float(r)))

    clips: dict[int, ClipFit] = {}
    for clip, rows in per_clip.items():
        errs = np.array([abs(r) for _s, r in rows], dtype=np.float64)
        intercept = intercepts[clip]
        clips[clip] = ClipFit(
            clip=clip,
            intercept_seconds=intercept,
            start_time=epoch + timedelta(seconds=intercept),
            n_samples=len(rows),
            n_inliers=sum(1 for s, _r in rows if s.name in inlier_names),
            residual_mad=float(np.median(errs)),
            max_abs_residual=float(errs.max()),
        )
    return clips


def _borrow_intercepts(clips: dict[int, ClipFit], all_keys: Sequence[FrameKey],
                       epoch: datetime, slope: float) -> dict[int, ClipFit]:
    """Place clips that produced no readable sample, using the filename clock.

    Leaving them out would reintroduce exactly the bug this module exists to
    remove: their frames would fall back to filename times while every other
    frame uses the fitted clock, putting two time bases in one folder.

    The filename clock is a valid *relative* clock within a clip -- only its
    absolute offset is suspect, and that offset is usually constant (filenames
    come from the NVR export, the overlay from the camera's own RTC). So we
    measure the offset on the clips that did fit and carry it across.
    """
    by_clip: dict[int, list[FrameKey]] = {}
    for key in all_keys:
        if key.filename_time is not None:
            by_clip.setdefault(key.clip, []).append(key)

    missing = [c for c in by_clip if c not in clips]
    if not missing or not clips:
        return clips

    def filename_intercept(clip: int) -> float | None:
        keys = by_clip.get(clip)
        if not keys:
            return None
        y = np.array([(k.filename_time - epoch).total_seconds() for k in keys])
        x = np.array([float(k.x) for k in keys])
        return float(np.median(y - slope * x))

    offsets = []
    for clip, fit in clips.items():
        base = filename_intercept(clip)
        if base is not None:
            offsets.append(fit.intercept_seconds - base)
    if not offsets:
        return clips
    offset = float(np.median(offsets))

    result = dict(clips)
    for clip in missing:
        base = filename_intercept(clip)
        if base is None:
            continue
        intercept = base + offset
        result[clip] = ClipFit(
            clip=clip, intercept_seconds=intercept,
            start_time=epoch + timedelta(seconds=intercept),
            n_samples=0, n_inliers=0, residual_mad=0.0, max_abs_residual=0.0,
            fitted=False)
    return result


def _with_warnings(fit: TimelineFit, x_kind: str) -> TimelineFit:
    """Attach human-readable warnings. Only a backwards slope is fatal."""
    warnings: list[str] = []

    if fit.slope <= 0:
        warnings.append(
            "Time runs backwards across the frames - the readings could not be "
            "fitted to a clock.")
    elif fit.implied_fps is not None:
        low, high = config.TIMELINE_PLAUSIBLE_FPS_RANGE
        if not (low <= fit.implied_fps <= high):
            warnings.append(
                f"Implied frame rate {fit.implied_fps:.2f} fps is outside the "
                f"plausible {low:g}-{high:g} fps range.")

    if fit.n_samples and fit.inlier_fraction < config.TIMELINE_MIN_INLIER_FRACTION:
        warnings.append(
            f"Only {fit.inlier_fraction:.0%} of readings fit a straight line "
            f"({fit.n_inliers} of {fit.n_samples}).")

    if fit.residual_mad > config.TIMELINE_WARN_RESIDUAL_MAD_SECONDS:
        warnings.append(
            f"Readings typically disagree with the fitted clock by "
            f"{fit.residual_mad:.1f} s.")

    if fit.max_abs_residual > config.TIMELINE_WARN_MAX_RESIDUAL_SECONDS:
        # A small MAD with a large worst-case is the signature of a piecewise
        # timeline: variable frame rate, dropped frames, or the clock being
        # adjusted mid-recording. One straight line genuinely cannot describe
        # that, and it is the one failure the residual spread alone hides.
        warnings.append(
            f"Worst reading is {fit.max_abs_residual:.1f} s off the fitted clock - "
            f"the recording may have gaps or a variable frame rate.")

    thin = [c.clip for c in fit.clips.values()
            if c.fitted and c.n_samples < config.TIMELINE_MIN_SAMPLES_FOR_CONFIDENT_CLIP]
    if thin:
        warnings.append(
            f"Clip(s) {', '.join(map(str, sorted(thin)))} have very few readings - "
            f"their start times are unverified.")

    borrowed = [c.clip for c in fit.clips.values() if not c.fitted]
    if borrowed:
        warnings.append(
            f"Clip(s) {', '.join(map(str, sorted(borrowed)))} had no readable clock; "
            f"their start times were derived from the other clips' offset.")

    if x_kind == X_ORDINAL:
        warnings.append(
            "Filenames carry no frame index, so frames are assumed evenly spaced.")

    return TimelineFit(
        slope=fit.slope, x_kind=fit.x_kind, epoch=fit.epoch, clips=fit.clips,
        n_samples=fit.n_samples, n_inliers=fit.n_inliers,
        residual_mad=fit.residual_mad, max_abs_residual=fit.max_abs_residual,
        implied_fps=fit.implied_fps, ok=fit.ok, warnings=tuple(warnings),
        residuals=fit.residuals, inliers=fit.inliers)


# --- applying ----------------------------------------------------------------

def apply_fit(
    fit: TimelineFit,
    keys: Sequence[FrameKey],
    quantum: float = config.TIMELINE_CLOCK_QUANTUM_SECONDS,
) -> dict[str, datetime]:
    """Timestamp every frame from the fitted line.

    Half the display quantum is added back: the overlay shows truncated whole
    seconds, so readings sit on average half a second behind the true time,
    and that bias would otherwise be baked into every frame.

    Frames belonging to a clip with no fit are omitted rather than guessed.
    Results carry sub-second precision, which is finer than the whole-second
    timestamps encoded in filenames today.
    """
    out: dict[str, datetime] = {}
    bias = quantum / 2.0
    for key in keys:
        clip = fit.clips.get(key.clip)
        if clip is None:
            continue
        seconds = (fit.slope * key.x + clip.intercept_seconds
                   + clip.user_offset_seconds + bias)
        out[key.name] = fit.epoch + timedelta(seconds=seconds)
    return out


def nudge(fit: TimelineFit, seconds: float, clip: int | None = None) -> TimelineFit:
    """Return a copy with a manual offset applied to one clip, or all of them.

    Kept out of the dialog so the adjustment is testable and so re-applying a
    stored offset on a later run goes through exactly the same code path.
    """
    clips = {
        cid: (replace(c, user_offset_seconds=c.user_offset_seconds + seconds)
              if clip is None or cid == clip else c)
        for cid, c in fit.clips.items()
    }
    return TimelineFit(
        slope=fit.slope, x_kind=fit.x_kind, epoch=fit.epoch, clips=clips,
        n_samples=fit.n_samples, n_inliers=fit.n_inliers,
        residual_mad=fit.residual_mad, max_abs_residual=fit.max_abs_residual,
        implied_fps=fit.implied_fps, ok=fit.ok, warnings=fit.warnings,
        residuals=fit.residuals, inliers=fit.inliers)


# --- sidecar document --------------------------------------------------------

SIDECAR_VERSION = 2


def build_document(fit: TimelineFit, samples: Sequence[TimeSample],
                   frames: dict[str, datetime],
                   confirmed_by_user: bool = False) -> dict:
    """Assemble the JSON-serializable sidecar body (see ``timestamp_ocr``).

    The ``samples`` block is what lets a review be reopened later without
    re-running any OCR -- previous readings, candidates and manual edits all
    survive in the file.
    """
    return {
        "version": SIDECAR_VERSION,
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "source": "timeline",
        "confirmed_by_user": bool(confirmed_by_user),
        "x_kind": fit.x_kind,
        "slope_seconds_per_x": fit.slope,
        "implied_fps": fit.implied_fps,
        "epoch": fit.epoch.isoformat(),
        "ok": fit.ok,
        "stats": {
            "n_frames": len(frames),
            "n_samples": fit.n_samples,
            "n_inliers": fit.n_inliers,
            "residual_mad": fit.residual_mad,
            "max_abs_residual": fit.max_abs_residual,
        },
        "warnings": list(fit.warnings),
        "clips": [
            {
                "clip": c.clip,
                "start_time": c.start_time.isoformat(),
                "intercept_seconds": c.intercept_seconds,
                "n_samples": c.n_samples,
                "n_inliers": c.n_inliers,
                "residual_mad": c.residual_mad,
                "max_abs_residual": c.max_abs_residual,
                "user_offset_seconds": c.user_offset_seconds,
                "fitted": c.fitted,
            }
            for c in sorted(fit.clips.values(), key=lambda c: c.clip)
        ],
        "samples": [
            {
                "name": s.name,
                "clip": s.clip,
                "x": s.x,
                "chosen_time": s.chosen.isoformat() if s.chosen else None,
                "filename_time": s.filename_time.isoformat() if s.filename_time else None,
                "edited": s.edited,
                "ignored": s.ignored,
                "inlier": s.name in fit.inliers,
                "residual_seconds": fit.residuals.get(s.name),
                "candidates": [
                    {
                        "text": getattr(c, "text", ""),
                        "time": (c.timestamp.isoformat()
                                 if getattr(c, "timestamp", None) else None),
                        "conf": float(getattr(c, "confidence", 0.0)),
                    }
                    for c in s.candidates
                ],
            }
            for s in samples
        ],
        "frames": {name: ts.isoformat() for name, ts in sorted(frames.items())},
    }
