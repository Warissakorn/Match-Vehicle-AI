"""Tests for the timestamp timeline fit.

This is where the accuracy of the whole "fix the times" feature lives, so it
gets the bulk of the coverage. Everything here is synthetic: the fit's input
is ``(clip, x, datetime)`` triples, so a known ground-truth line can be
generated in a helper and the recovered parameters compared against it -- no
images, no easyocr, no GPU, which is what lets these run in CI (numpy/scipy/
pytest only).

The cases that matter most, and would be easy to lose in a refactor:
  * outlier rejection actually working (``test_endpoint_outlier_*``)
  * the MAD-of-zero tolerance floor (``test_perfect_data_keeps_all_inliers``)
  * a low MAD with a large worst-case residual being flagged, since that is
    the piecewise/variable-frame-rate case a single line cannot represent
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from mash_reid import timeline
from mash_reid.timeline import FrameKey, TimeSample

EPOCH = datetime(2026, 7, 23, 10, 0, 0)


def _samples(slope=1 / 25.0, starts=(0.0,), n=20, step=25, quantize=True,
             outliers=(), base=EPOCH):
    """Build TimeSamples along a known line.

    ``outliers`` is a sequence of ``(index_within_all_samples, delta_seconds)``.
    ``quantize`` truncates to whole seconds, matching a real overlay clock.
    """
    out = []
    idx = 0
    bad = dict(outliers)
    for clip, start in enumerate(starts):
        for i in range(n):
            x = i * step
            seconds = slope * x + start
            if quantize:
                seconds = float(int(seconds))
            seconds += bad.get(idx, 0.0)
            out.append(TimeSample(
                name=f"A_20260723_100000_c{clip:02d}_{x:06d}.jpg",
                clip=clip, x=x, chosen=base + timedelta(seconds=seconds)))
            idx += 1
    return out


# --- parse_frame_key ---------------------------------------------------------

def test_parse_single_clip_name():
    key = timeline.parse_frame_key("A_20260723_101530_000123.jpg")
    assert (key.clip, key.x) == (0, 123)
    assert key.filename_time == datetime(2026, 7, 23, 10, 15, 30)


def test_parse_multi_clip_name():
    key = timeline.parse_frame_key("A_20260723_101530_c01_000123.jpg")
    assert (key.clip, key.x) == (1, 123)


def test_parse_index_wider_than_six_digits():
    # %06d is a minimum width; past ~9 hours at 30fps the field grows.
    key = timeline.parse_frame_key("A_20260723_101530_c01_1234567.jpg")
    assert key.x == 1234567


def test_parse_clip_wider_than_two_digits():
    key = timeline.parse_frame_key("A_20260723_101530_c100_000123.jpg")
    assert key.clip == 100


def test_parse_point_containing_underscore():
    key = timeline.parse_frame_key("cam_north_20260723_101530_000123.jpg")
    assert (key.clip, key.x) == (0, 123)


def test_parse_png_extension():
    assert timeline.parse_frame_key("B_20260723_101530_000001.png") is not None


def test_parse_foreign_name_is_none():
    assert timeline.parse_frame_key("random_photo.jpg") is None


def test_parse_name_without_index_is_none():
    assert timeline.parse_frame_key("cam1-20260723_101530.png") is None


def test_parse_impossible_date_keeps_index_but_no_time():
    key = timeline.parse_frame_key("A_20261332_101530_000123.jpg")
    assert key is not None and key.x == 123
    assert key.filename_time is None


# --- build_keys --------------------------------------------------------------

def test_build_keys_uses_frame_index_when_all_parse():
    names = [f"A_20260723_101530_{i:06d}.jpg" for i in range(0, 100, 25)]
    keys, kind, unparsed = timeline.build_keys(names)
    assert kind == timeline.X_FRAME_INDEX
    assert [k.x for k in keys] == [0, 25, 50, 75]
    assert unparsed == []


def test_build_keys_tolerates_a_few_foreign_names():
    names = [f"A_20260723_101530_{i:06d}.jpg" for i in range(100)] + ["notes.jpg"]
    keys, kind, unparsed = timeline.build_keys(names)
    assert kind == timeline.X_FRAME_INDEX
    assert unparsed == ["notes.jpg"]
    assert len(keys) == 100  # the foreign name is excluded from the axis


def test_build_keys_falls_back_to_ordinal_when_mostly_foreign():
    names = ["a.jpg", "b.jpg", "c.jpg", "A_20260723_101530_000001.jpg"]
    keys, kind, _unparsed = timeline.build_keys(names)
    assert kind == timeline.X_ORDINAL
    assert [k.x for k in keys] == [0, 1, 2, 3]
    assert all(k.clip == 0 for k in keys)


def test_build_keys_empty():
    keys, kind, unparsed = timeline.build_keys([])
    assert keys == [] and unparsed == [] and kind == timeline.X_ORDINAL


def test_build_keys_single_frame_index_falls_back():
    # One parseable name can't establish an axis (needs >= 2).
    keys, kind, _ = timeline.build_keys(["A_20260723_101530_000001.jpg"])
    assert kind == timeline.X_ORDINAL


# --- select_samples ----------------------------------------------------------

def _keys(n, clips=1, step=25):
    return [FrameKey(name=f"f_c{c}_{i:06d}.jpg", clip=c, x=i * step)
            for c in range(clips) for i in range(n)]


def test_select_samples_returns_everything_when_small():
    keys = _keys(5)
    assert len(timeline.select_samples(keys, total_target=40)) == 5


def test_select_samples_includes_both_endpoints():
    keys = _keys(1000)
    picked = timeline.select_samples(keys, total_target=40)
    xs = [k.x for k in picked]
    assert min(xs) == 0 and max(xs) == 999 * 25


def test_select_samples_honours_per_clip_floor():
    # 12 clips x 40-sample target would be ~3 each; the floor lifts it.
    keys = _keys(200, clips=12)
    picked = timeline.select_samples(keys, total_target=40, min_per_clip=8)
    per_clip = {}
    for k in picked:
        per_clip[k.clip] = per_clip.get(k.clip, 0) + 1
    assert all(count >= 8 for count in per_clip.values())
    assert len(per_clip) == 12


def test_select_samples_is_deterministic():
    keys = _keys(500, clips=3)
    assert timeline.select_samples(keys) == timeline.select_samples(keys)


def test_select_samples_single_frame_clip():
    keys = [FrameKey(name="a.jpg", clip=0, x=0)]
    assert timeline.select_samples(keys) == keys


def test_select_samples_empty():
    assert timeline.select_samples([]) == []


# --- theil_sen ---------------------------------------------------------------

def test_theil_sen_exact_on_a_clean_line():
    x = np.arange(10, dtype=float)
    y = 3.0 * x + 7.0
    slope, intercept = timeline.theil_sen(x, y)
    assert slope == pytest.approx(3.0)
    assert intercept == pytest.approx(7.0)


def test_theil_sen_ignores_duplicate_x_pairs():
    slope, _ = timeline.theil_sen([1.0, 1.0, 2.0], [5.0, 5.0, 7.0])
    assert slope == pytest.approx(2.0)


def test_theil_sen_all_identical_x_is_not_a_crash():
    slope, intercept = timeline.theil_sen([4.0, 4.0], [1.0, 3.0])
    assert slope == 0.0 and intercept == pytest.approx(2.0)


def test_theil_sen_empty():
    assert timeline.theil_sen([], []) == (0.0, 0.0)


# --- fit_timeline: recovery --------------------------------------------------

def test_recovers_slope_and_start_exactly_without_quantization():
    fit = timeline.fit_timeline(_samples(quantize=False))
    assert fit.ok
    assert fit.slope == pytest.approx(1 / 25.0, rel=1e-9)
    assert fit.clips[0].start_time == EPOCH


def test_recovers_slope_within_tolerance_with_whole_second_clock():
    fit = timeline.fit_timeline(_samples(n=40, quantize=True))
    assert fit.ok
    assert fit.slope == pytest.approx(1 / 25.0, rel=1e-3)
    assert abs((fit.clips[0].start_time - EPOCH).total_seconds()) <= 1.0


def test_implied_fps_matches_the_generated_rate():
    # 40 samples spread over a 3.3 h clip -- the real shape, since
    # select_samples spans the whole clip rather than bunching.
    fit = timeline.fit_timeline(
        _samples(slope=1 / 29.97, n=40, step=9000, quantize=True))
    assert fit.implied_fps == pytest.approx(29.97, rel=1e-3)


def test_slope_precision_is_bounded_by_span_not_sample_count():
    # Worth knowing when tuning sample counts: the 1 s clock quantum divides
    # by the *time span*, not the number of samples. 400 samples bunched into
    # 400 s resolve the rate worse than 40 samples spread over 3 hours -- so
    # spreading samples matters far more than taking more of them.
    bunched = timeline.fit_timeline(
        _samples(slope=1 / 29.97, n=400, step=30, quantize=True))
    spread = timeline.fit_timeline(
        _samples(slope=1 / 29.97, n=40, step=9000, quantize=True))
    assert abs(spread.implied_fps - 29.97) < abs(bunched.implied_fps - 29.97)


# --- fit_timeline: outlier rejection (the whole point) -----------------------

def test_endpoint_outlier_does_not_move_the_fit():
    # One hour-digit misread on the last frame -- the classic OCR failure.
    good = timeline.fit_timeline(_samples(n=40, quantize=False))
    bad = timeline.fit_timeline(_samples(n=40, quantize=False, outliers=[(39, 3600.0)]))
    assert bad.slope == pytest.approx(good.slope, rel=1e-6)


def test_endpoint_outlier_would_wreck_least_squares():
    # Documents *why* Theil-Sen was chosen. If someone "simplifies" the
    # estimator back to np.polyfit, this test is what fails.
    samples = _samples(n=40, quantize=False, outliers=[(39, 3600.0)])
    x = np.array([s.x for s in samples], dtype=float)
    y = np.array([(s.chosen - EPOCH).total_seconds() for s in samples])
    ls_slope = np.polyfit(x, y, 1)[0]
    ts_slope = timeline.fit_timeline(samples).slope
    assert abs(ts_slope - 1 / 25.0) < 1e-6
    assert abs(ls_slope - 1 / 25.0) > 1e-3  # least squares is dragged off


def test_survives_outliers_just_under_the_breakdown_point():
    # 11 of 40 is ~27%, under Theil-Sen's 29.3% breakdown.
    outliers = [(i, 3600.0 if i % 2 else -3600.0) for i in range(11)]
    fit = timeline.fit_timeline(_samples(n=40, quantize=False, outliers=outliers))
    assert fit.slope == pytest.approx(1 / 25.0, rel=1e-6)
    assert fit.n_inliers == 29


def test_too_many_outliers_is_reported_not_crashed():
    outliers = [(i, 3600.0) for i in range(30)]
    fit = timeline.fit_timeline(_samples(n=40, quantize=False, outliers=outliers))
    assert isinstance(fit.warnings, tuple)
    assert fit.inlier_fraction < 1.0


def test_realistic_gallery_stays_accurate_from_a_small_sample():
    # The shape of the real complaint: 12,557 frames at 29.97fps, sampled,
    # with 8% gross misreads. Every frame must land within a second.
    rng = np.random.default_rng(0)
    n, fps, step = 12557, 29.97, 30
    keys = [FrameKey(name=f"A_20260723_100000_{i * step:06d}.jpg", clip=0, x=i * step)
            for i in range(n)]
    picked = timeline.select_samples(keys, total_target=40)

    samples = []
    for j, k in enumerate(picked):
        seconds = float(int(k.x / fps))  # whole-second overlay
        if rng.random() < 0.08:
            seconds += rng.choice([-3600.0, 3600.0, 600.0, -60.0])
        samples.append(TimeSample(name=k.name, clip=0, x=k.x,
                                  chosen=EPOCH + timedelta(seconds=seconds)))

    fit = timeline.fit_timeline(samples)
    assert fit.ok
    times = timeline.apply_fit(fit, keys)
    assert len(times) == n
    errors = [abs((times[k.name] - (EPOCH + timedelta(seconds=k.x / fps))).total_seconds())
              for k in keys]
    assert max(errors) < 1.0


# --- fit_timeline: the MAD-of-zero trap --------------------------------------

def test_perfect_data_keeps_all_inliers():
    # Regression guard for the tolerance floor. With perfectly consistent
    # readings the MAD is exactly 0; without a floor the tolerance collapses
    # to 0, every sample becomes an "outlier", and the refit runs on nothing.
    fit = timeline.fit_timeline(_samples(n=30, quantize=False))
    assert fit.residual_mad == pytest.approx(0.0, abs=1e-9)
    assert fit.inlier_fraction == 1.0
    assert fit.n_inliers == 30


def test_quantized_data_keeps_all_inliers():
    # Same trap via the realistic path: whole-second readings make many
    # residuals identical, so the MAD is tiny but non-zero.
    fit = timeline.fit_timeline(_samples(n=30, quantize=True))
    assert fit.inlier_fraction == 1.0


# --- fit_timeline: multiple clips -------------------------------------------

def test_shared_slope_with_separate_starts():
    fit = timeline.fit_timeline(
        _samples(starts=(0.0, 5000.0, 9000.0), n=10, quantize=False))
    assert fit.slope == pytest.approx(1 / 25.0, rel=1e-9)
    assert len(fit.clips) == 3
    for clip, start in zip(range(3), (0.0, 5000.0, 9000.0)):
        assert fit.clips[clip].start_time == EPOCH + timedelta(seconds=start)


def test_thin_clip_still_placed_and_warned():
    samples = _samples(starts=(0.0,), n=20, quantize=False)
    samples.append(TimeSample(name="A_20260723_100000_c01_000100.jpg", clip=1,
                              x=100, chosen=EPOCH + timedelta(seconds=8004.0)))
    fit = timeline.fit_timeline(samples)
    assert 1 in fit.clips
    assert fit.clips[1].n_samples == 1
    assert any("unverified" in w for w in fit.warnings)


def test_clip_with_no_readings_borrows_the_offset():
    # Clip 1 produced no readable clock. Its frames must still be placed, or
    # they fall back to filename times and reintroduce two clocks in one
    # folder -- the exact bug the fit exists to remove.
    offset = 120.0  # overlay clock runs 2 minutes ahead of the filename clock
    samples, all_keys = [], []
    for clip, start in enumerate((0.0, 4000.0)):
        for i in range(10):
            x = i * 25
            name = f"A_20260723_100000_c{clip:02d}_{x:06d}.jpg"
            fname_time = EPOCH + timedelta(seconds=x / 25.0 + start)
            all_keys.append(FrameKey(name=name, clip=clip, x=x,
                                     filename_time=fname_time))
            if clip == 0:
                samples.append(TimeSample(
                    name=name, clip=clip, x=x,
                    chosen=fname_time + timedelta(seconds=offset)))

    fit = timeline.fit_timeline(samples, all_keys=all_keys)
    assert 1 in fit.clips
    assert fit.clips[1].fitted is False
    expected = EPOCH + timedelta(seconds=4000.0 + offset)
    assert abs((fit.clips[1].start_time - expected).total_seconds()) < 1.0
    assert any("no readable clock" in w for w in fit.warnings)


def test_clip_with_no_readings_is_omitted_without_all_keys():
    fit = timeline.fit_timeline(_samples(n=10, quantize=False))
    assert set(fit.clips) == {0}


# --- fit_timeline: degenerate input -----------------------------------------

def test_no_samples_is_not_ok():
    fit = timeline.fit_timeline([])
    assert fit.ok is False
    assert "readable" in fit.warnings[0]


def test_all_unreadable_samples_is_not_ok():
    samples = [TimeSample(name="a.jpg", clip=0, x=0, chosen=None)]
    assert timeline.fit_timeline(samples).ok is False


def test_ignored_samples_are_excluded():
    samples = _samples(n=10, quantize=False)
    marked = [TimeSample(**{**s.__dict__, "ignored": True}) if i < 9 else s
              for i, s in enumerate(samples)]
    fit = timeline.fit_timeline(marked)
    assert fit.n_samples == 1


def test_single_sample_cannot_establish_a_rate():
    fit = timeline.fit_timeline(_samples(n=1, quantize=False))
    assert fit.ok is False
    assert any("rate" in w for w in fit.warnings)


def test_all_identical_x_is_not_a_crash():
    samples = [TimeSample(name=f"{i}.jpg", clip=0, x=7,
                          chosen=EPOCH + timedelta(seconds=i)) for i in range(4)]
    fit = timeline.fit_timeline(samples)
    assert fit.ok is False


def test_backwards_time_is_flagged_fatal():
    fit = timeline.fit_timeline(_samples(slope=-1 / 25.0, n=20, quantize=False))
    assert fit.ok is False
    assert any("backwards" in w for w in fit.warnings)


def test_implausible_frame_rate_is_warned():
    # 10 s per source frame -> 0.1 fps, far outside the plausible range.
    fit = timeline.fit_timeline(_samples(slope=10.0, n=20, quantize=False))
    assert any("plausible" in w for w in fit.warnings)


def test_piecewise_timeline_is_caught_by_max_residual():
    # A clock adjustment halfway through: most readings agree beautifully
    # (small MAD) but the worst one is far off. This is the failure a single
    # straight line genuinely cannot describe, and MAD alone would hide it.
    samples = _samples(n=40, quantize=False,
                       outliers=[(i, 60.0) for i in range(20, 40)])
    fit = timeline.fit_timeline(samples)
    assert fit.max_abs_residual > config_warn()
    assert any("gaps or a variable frame rate" in w for w in fit.warnings)


def config_warn():
    import config
    return config.TIMELINE_WARN_MAX_RESIDUAL_SECONDS


def test_ordinal_axis_reports_no_fps_and_warns():
    samples = _samples(n=20, quantize=False)
    fit = timeline.fit_timeline(samples, x_kind=timeline.X_ORDINAL)
    assert fit.implied_fps is None
    assert any("evenly spaced" in w for w in fit.warnings)


def test_fit_is_deterministic_under_reordering():
    samples = _samples(n=20, quantize=True)
    a = timeline.fit_timeline(samples)
    b = timeline.fit_timeline(list(reversed(samples)))
    assert a.slope == b.slope
    assert a.clips[0].intercept_seconds == b.clips[0].intercept_seconds


# --- apply_fit ---------------------------------------------------------------

def test_apply_is_strictly_increasing_within_a_clip():
    fit = timeline.fit_timeline(_samples(n=20, quantize=False))
    keys = [FrameKey(name=f"f{i:06d}.jpg", clip=0, x=i * 25) for i in range(50)]
    times = timeline.apply_fit(fit, keys)
    ordered = [times[k.name] for k in keys]
    assert all(b > a for a, b in zip(ordered, ordered[1:]))


def test_apply_compensates_the_half_second_truncation_bias():
    fit = timeline.fit_timeline(_samples(n=20, quantize=False))
    keys = [FrameKey(name="f.jpg", clip=0, x=0)]
    times = timeline.apply_fit(fit, keys)
    assert (times["f.jpg"] - EPOCH).total_seconds() == pytest.approx(0.5)


def test_apply_yields_sub_second_precision():
    fit = timeline.fit_timeline(_samples(slope=1 / 30.0, n=20, quantize=False))
    keys = [FrameKey(name=f"f{i}.jpg", clip=0, x=i) for i in range(3)]
    times = timeline.apply_fit(fit, keys)
    assert any(t.microsecond for t in times.values())


def test_apply_skips_frames_from_an_unfitted_clip():
    fit = timeline.fit_timeline(_samples(n=10, quantize=False))
    keys = [FrameKey(name="known.jpg", clip=0, x=0),
            FrameKey(name="orphan.jpg", clip=9, x=0)]
    assert set(timeline.apply_fit(fit, keys)) == {"known.jpg"}


def test_apply_then_refit_round_trips():
    fit = timeline.fit_timeline(_samples(n=20, quantize=False))
    keys = [FrameKey(name=f"f{i:06d}.jpg", clip=0, x=i * 25) for i in range(20)]
    times = timeline.apply_fit(fit, keys)
    again = timeline.fit_timeline(
        [TimeSample(name=k.name, clip=k.clip, x=k.x, chosen=times[k.name]) for k in keys])
    assert again.slope == pytest.approx(fit.slope, rel=1e-9)


# --- nudge -------------------------------------------------------------------

def test_nudge_all_clips_shifts_every_frame():
    fit = timeline.fit_timeline(_samples(starts=(0.0, 5000.0), n=10, quantize=False))
    keys = [FrameKey(name=f"c{c}.jpg", clip=c, x=0) for c in (0, 1)]
    before = timeline.apply_fit(fit, keys)
    after = timeline.apply_fit(timeline.nudge(fit, 30.0), keys)
    for k in keys:
        assert (after[k.name] - before[k.name]).total_seconds() == pytest.approx(30.0)


def test_nudge_one_clip_leaves_the_others_alone():
    fit = timeline.fit_timeline(_samples(starts=(0.0, 5000.0), n=10, quantize=False))
    keys = [FrameKey(name=f"c{c}.jpg", clip=c, x=0) for c in (0, 1)]
    before = timeline.apply_fit(fit, keys)
    after = timeline.apply_fit(timeline.nudge(fit, 30.0, clip=1), keys)
    assert after["c0.jpg"] == before["c0.jpg"]
    assert (after["c1.jpg"] - before["c1.jpg"]).total_seconds() == pytest.approx(30.0)


# --- build_document ----------------------------------------------------------

def test_document_is_json_serializable():
    import json

    samples = _samples(n=10, quantize=False)
    fit = timeline.fit_timeline(samples)
    keys = [FrameKey(name=s.name, clip=s.clip, x=s.x) for s in samples]
    doc = timeline.build_document(fit, samples, timeline.apply_fit(fit, keys))
    json.dumps(doc)  # must not raise


def test_document_carries_every_frame_and_the_fit():
    samples = _samples(n=10, quantize=False)
    fit = timeline.fit_timeline(samples)
    keys = [FrameKey(name=s.name, clip=s.clip, x=s.x) for s in samples]
    frames = timeline.apply_fit(fit, keys)
    doc = timeline.build_document(fit, samples, frames, confirmed_by_user=True)

    assert doc["version"] == timeline.SIDECAR_VERSION
    assert doc["source"] == "timeline"
    assert doc["confirmed_by_user"] is True
    assert len(doc["frames"]) == len(frames)
    assert len(doc["samples"]) == len(samples)
    assert doc["stats"]["n_samples"] == fit.n_samples
    assert doc["clips"][0]["clip"] == 0
