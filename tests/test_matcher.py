"""Tests for the matcher: cosine similarity, temporal gating, ranking.

These use synthetic embeddings so they run without any model or network.
"""

from datetime import datetime, timedelta

import numpy as np

import config
from mash_reid import matcher
from mash_reid.matcher import VehicleRecord


def _unit(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


def _rec(record_id, point, emb, t):
    base = datetime(2026, 7, 23, 10, 0, 0)
    return VehicleRecord(
        record_id=record_id,
        point=point,
        frame_path=f"{point}_{record_id}.jpg",
        timestamp=base + timedelta(seconds=t),
        bbox=(0, 0, 10, 10),
        confidence=0.9,
        embedding=_unit(emb),
    )


def test_cosine_identical_is_one():
    a = _unit([1, 2, 3])[None, :]
    sim = matcher.cosine_similarity_matrix(a, a)
    assert sim.shape == (1, 1)
    assert abs(sim[0, 0] - 1.0) < 1e-5


def test_cosine_orthogonal_is_zero():
    a = _unit([1, 0])[None, :]
    b = _unit([0, 1])[None, :]
    assert abs(matcher.cosine_similarity_matrix(a, b)[0, 0]) < 1e-5


def test_cosine_empty_inputs():
    empty = np.zeros((0, 4), dtype=np.float32)
    filled = np.zeros((3, 4), dtype=np.float32)
    assert matcher.cosine_similarity_matrix(empty, filled).shape == (0, 3)
    assert matcher.cosine_similarity_matrix(filled, empty).shape == (3, 0)


def test_time_gate_mask_window():
    ta = [datetime(2026, 7, 23, 10, 0, 0)]
    tb = [
        datetime(2026, 7, 23, 9, 59, 0),   # before A -> excluded
        datetime(2026, 7, 23, 10, 1, 0),   # +60s -> inside
        datetime(2026, 7, 23, 10, 20, 0),  # +1200s -> outside
    ]
    mask = matcher.time_gate_mask(ta, tb, min_travel_seconds=0, max_travel_seconds=600)
    assert list(mask[0]) == [False, True, False]


def test_match_picks_most_similar_within_window():
    # A vehicle appears at A (t=0). Its true twin is at B (t=60) with same embed.
    a = [_rec(0, "A", [1, 0, 0], t=0)]
    b = [
        _rec(0, "B", [0, 1, 0], t=60),   # different look
        _rec(1, "B", [1, 0, 0], t=60),   # same look, in window  <-- expected
        _rec(2, "B", [1, 0, 0], t=5000), # same look but too late -> gated out
    ]
    cfg = config.MatchConfig(similarity_threshold=0.5, use_time_gate=True,
                             min_travel_seconds=0, max_travel_seconds=600)
    results = matcher.match(a, b, cfg)
    assert len(results) == 1
    best = results[0].best
    assert best is not None
    assert best.b_record_id == 1
    assert best.similarity > 0.99


def test_threshold_filters_out_weak_matches():
    a = [_rec(0, "A", [1, 0], t=0)]
    b = [_rec(0, "B", [0, 1], t=60)]  # orthogonal -> sim ~0
    cfg = config.MatchConfig(similarity_threshold=0.5, use_time_gate=False)
    results = matcher.match(a, b, cfg)
    assert results[0].best is None


def test_time_gate_excludes_b_before_a():
    a = [_rec(0, "A", [1, 0], t=100)]
    b = [_rec(0, "B", [1, 0], t=0)]  # B happened before A
    cfg = config.MatchConfig(similarity_threshold=0.5, use_time_gate=True,
                             min_travel_seconds=0, max_travel_seconds=600)
    assert matcher.match(a, b, cfg)[0].best is None


def test_one_to_one_assignment_is_unique():
    # Two A vehicles, two identical-looking B vehicles; one-to-one must not
    # assign both A's to the same B.
    a = [_rec(0, "A", [1, 0], t=0), _rec(1, "A", [1, 0], t=0)]
    b = [_rec(0, "B", [1, 0], t=60), _rec(1, "B", [1, 0], t=60)]
    cfg = config.MatchConfig(similarity_threshold=0.5, use_time_gate=False, one_to_one=True)
    results = matcher.match(a, b, cfg)
    assigned = [r.best.b_record_id for r in results if r.best]
    assert sorted(assigned) == [0, 1]


def test_empty_gallery_yields_no_matches():
    a = [_rec(0, "A", [1, 0], t=0)]
    results = matcher.match(a, [], config.MatchConfig())
    assert len(results) == 1
    assert results[0].best is None
