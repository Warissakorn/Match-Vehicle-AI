"""Equivalence tests for the chunked/streaming matcher.

``matcher.match`` and ``matcher.cluster_same_point`` compute their similarity
matrices one row-block at a time so peak memory stays bounded on real gallery
sizes (3,670 x 12,557). That is an allocation-shape change only: these tests
pin the streaming output to straightforward dense reference implementations,
with ``_ROW_BLOCK`` patched down to single digits so every test actually
exercises many blocks.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

import config
from mash_reid import matcher
from mash_reid.matcher import MatchCandidate, MatchResult, VehicleRecord

_BASE_TIME = datetime(2024, 1, 1, 12, 0, 0)
_DIM = 16


def _make_records(n: int, seed: int, point: str = "A") -> list[VehicleRecord]:
    """n records with L2-normalized random embeddings spread over one hour."""
    rng = np.random.default_rng(seed)
    emb = rng.normal(size=(n, _DIM))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return [
        VehicleRecord(
            record_id=i,
            point=point,
            frame_path=f"{point}{i}.jpg",
            timestamp=_BASE_TIME + timedelta(seconds=float(rng.integers(0, 3600))),
            bbox=(0, 0, 10, 10),
            confidence=0.9,
            embedding=emb[i].astype(np.float32),
            timestamp_source="timeline",
        )
        for i in range(n)
    ]


@pytest.fixture
def tiny_blocks(monkeypatch):
    """Force multi-block execution even for toy-sized inputs."""
    monkeypatch.setattr(matcher, "_ROW_BLOCK", 9)


# --- dense reference implementations (deliberately naive) ---------------------


def _reference_match(records_a, records_b, cfg) -> list[MatchResult]:
    na, nb = len(records_a), len(records_b)
    if na == 0:
        return []
    if nb == 0:
        return [MatchResult(a_record_id=r.record_id, candidates=[]) for r in records_a]

    emb_a = np.stack([r.embedding for r in records_a]).astype(np.float32)
    emb_b = np.stack([r.embedding for r in records_b]).astype(np.float32)

    # Pure-Python gate: independent of matcher's vectorized helpers.
    sim = (emb_a @ emb_b.T).astype(np.float64)
    for i in range(na):
        for j in range(nb):
            if not cfg.use_time_gate:
                continue
            delta = (records_b[j].timestamp - records_a[i].timestamp).total_seconds()
            if not (cfg.min_travel_seconds <= delta <= cfg.max_travel_seconds):
                sim[i, j] = -np.inf

    if cfg.one_to_one:
        from scipy.optimize import linear_sum_assignment

        big = 1e6
        cost = np.where(np.isfinite(sim), -sim, big)
        rows, cols = linear_sum_assignment(cost)
        assigned = {int(r): int(c) for r, c in zip(rows, cols)}
        results = []
        for i, rec in enumerate(records_a):
            candidates: list[MatchCandidate] = []
            j = assigned.get(i)
            if j is not None:
                score = float(sim[i, j])
                if np.isfinite(score) and score >= cfg.similarity_threshold:
                    candidates.append(
                        MatchCandidate(b_record_id=records_b[j].record_id,
                                       similarity=score))
            results.append(MatchResult(a_record_id=rec.record_id, candidates=candidates))
        return results

    results = []
    for i, rec in enumerate(records_a):
        scored = [(float(sim[i, j]), j) for j in range(nb)]
        kept = [(s, j) for s, j in scored
                if np.isfinite(s) and s >= cfg.similarity_threshold]
        # "Best first" is descending similarity; the j tiebreak matches what
        # np.argsort(-row) does for the exact ties that never occur here.
        kept.sort(key=lambda t: (-t[0], t[1]))
        candidates = [MatchCandidate(b_record_id=records_b[j].record_id, similarity=s)
                      for s, j in kept[: cfg.top_k]]
        results.append(MatchResult(a_record_id=rec.record_id, candidates=candidates))
    return results


def _reference_clusters(records, threshold) -> dict[int, int]:
    n = len(records)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    emb = np.stack([r.embedding for r in records]).astype(np.float64)
    sim = emb @ emb.T
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                union(i, j)

    root_to_cluster: dict[int, int] = {}
    result: dict[int, int] = {}
    for i, rec in enumerate(records):
        root = find(i)
        if root not in root_to_cluster:
            root_to_cluster[root] = len(root_to_cluster)
        result[rec.record_id] = root_to_cluster[root]
    return result


def _assert_results_equal(got: list[MatchResult], want: list[MatchResult]) -> None:
    assert [r.a_record_id for r in got] == [r.a_record_id for r in want]
    for g, w in zip(got, want):
        assert [c.b_record_id for c in g.candidates] == \
               [c.b_record_id for c in w.candidates]
        for cg, cw in zip(g.candidates, w.candidates):
            assert cg.similarity == pytest.approx(cw.similarity, abs=1e-6)


# --- match() vs dense reference -----------------------------------------------


@pytest.mark.parametrize("one_to_one", [False, True])
@pytest.mark.parametrize("use_time_gate", [False, True])
def test_match_equals_dense_reference(tiny_blocks, one_to_one, use_time_gate):
    records_a = _make_records(23, seed=1)
    records_b = _make_records(31, seed=2, point="B")
    cfg = config.MatchConfig(
        top_k=5, use_time_gate=use_time_gate, one_to_one=one_to_one)
    got = matcher.match(records_a, records_b, cfg)
    want = _reference_match(records_a, records_b, cfg)
    _assert_results_equal(got, want)


def test_match_single_block_path_also_matches_reference():
    # Default _ROW_BLOCK (1024): the toy sizes here stay within one block.
    records_a = _make_records(7, seed=3)
    records_b = _make_records(11, seed=4, point="B")
    cfg = config.MatchConfig(top_k=3)
    got = matcher.match(records_a, records_b, cfg)
    want = _reference_match(records_a, records_b, cfg)
    _assert_results_equal(got, want)


def test_match_empty_a_returns_empty_list():
    assert matcher.match([], _make_records(4, seed=5), config.MatchConfig()) == []


def test_match_empty_b_gives_empty_candidates_per_a(tiny_blocks):
    records_a = _make_records(5, seed=6)
    got = matcher.match(records_a, [], config.MatchConfig())
    assert [r.a_record_id for r in got] == [r.record_id for r in records_a]
    assert all(r.candidates == [] for r in got)


def test_fully_gated_one_to_one_falls_back_and_reports_nothing(tiny_blocks):
    # min_travel far above any real delta gates every pair; with no finite
    # entry the Hungarian pass must be skipped and every A reports nothing --
    # not crash, and not a sentinel-cost partner.
    records_a = _make_records(6, seed=7)
    records_b = _make_records(8, seed=8, point="B")
    cfg = config.MatchConfig(one_to_one=True, min_travel_seconds=10_000.0)
    got = matcher.match(records_a, records_b, cfg)
    assert len(got) == len(records_a)
    assert all(r.candidates == [] for r in got)


def test_top_k_limits_candidates_even_across_blocks(tiny_blocks):
    # With block size 9 and 40 B-records, every row sees its full 40-wide
    # candidate set assembled block-by-block -- top_k must still cap at 2.
    records_a = _make_records(12, seed=9)
    records_b = _make_records(40, seed=10, point="B")
    cfg = config.MatchConfig(top_k=2, use_time_gate=False)
    got = matcher.match(records_a, records_b, cfg)
    assert all(len(r.candidates) <= 2 for r in got)
    _assert_results_equal(got, _reference_match(records_a, records_b, cfg))


def test_threshold_filters_weak_pairs_identically(tiny_blocks):
    # A threshold above most similarities exercises the per-block threshold
    # rejection path against the dense reference.
    records_a = _make_records(20, seed=11)
    records_b = _make_records(24, seed=12, point="B")
    cfg = config.MatchConfig(similarity_threshold=0.95, top_k=10,
                             use_time_gate=False)
    got = matcher.match(records_a, records_b, cfg)
    assert sum(len(r.candidates) for r in got) < 20 * 10  # some were dropped
    _assert_results_equal(got, _reference_match(records_a, records_b, cfg))


# --- cluster_same_point vs dense reference ------------------------------------


def _records_with_duplicates(n_unique: int, dup_counts: dict[int, int],
                             seed: int) -> list[VehicleRecord]:
    """Records where some embeddings are exact duplicates, forcing clusters."""
    base = _make_records(n_unique, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    records: list[VehicleRecord] = []
    next_id = n_unique
    for rec in base:
        records.append(rec)
        for _ in range(dup_counts.get(rec.record_id, 0)):
            records.append(VehicleRecord(
                record_id=next_id,
                point="A",
                frame_path=f"dup{next_id}.jpg",
                timestamp=_BASE_TIME + timedelta(seconds=float(rng.integers(0, 3600))),
                bbox=(0, 0, 10, 10),
                confidence=0.9,
                embedding=rec.embedding.copy(),
                timestamp_source="timeline",
            ))
            next_id += 1
    return records


def test_clusters_equal_dense_reference_with_duplicates(tiny_blocks):
    records = _records_with_duplicates(
        30, dup_counts={0: 3, 5: 2, 17: 4}, seed=13)
    got = matcher.cluster_same_point(records)
    want = _reference_clusters(records, config.DEFAULT_SAME_POINT_SIMILARITY_THRESHOLD)
    assert got == want
    # The duplicates really did collapse into shared clusters.
    by_cluster: dict[int, list[int]] = {}
    for rid, cid in got.items():
        by_cluster.setdefault(cid, []).append(rid)
    assert max(len(v) for v in by_cluster.values()) >= 4


def test_clusters_all_singletons_when_threshold_impossible(tiny_blocks):
    records = _make_records(15, seed=14)
    got = matcher.cluster_same_point(records, similarity_threshold=2.0)
    assert sorted(got.values()) == list(range(len(records)))


def test_cluster_ids_follow_first_seen_order(tiny_blocks):
    # Three distinct vehicles; the second record seen starts cluster 1, etc.,
    # regardless of how many blocks the pair scan needed.
    import dataclasses

    base = _make_records(3, seed=15)
    twin = dataclasses.replace(base[1], record_id=99)
    records = [base[1], twin, base[0], base[2]]
    got = matcher.cluster_same_point(records)
    assert got[base[1].record_id] == got[99]      # duplicate of index 0 -> 0
    assert got[base[0].record_id] == 1
    assert got[base[2].record_id] == 2


def test_cluster_empty_input():
    assert matcher.cluster_same_point([]) == {}


def test_blocks_partition_respects_row_block(monkeypatch):
    assert [pair for pair in matcher._blocks(0)] == []
    monkeypatch.setattr(matcher, "_ROW_BLOCK", 9)
    assert list(matcher._blocks(10)) == [(0, 9), (9, 10)]
    assert all(stop - start <= matcher._ROW_BLOCK
               for start, stop in matcher._blocks(5000))
