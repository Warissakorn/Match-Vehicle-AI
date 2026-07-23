"""Match vehicles seen at point A against vehicles seen at point B.

Given the appearance embeddings (and timestamps) of every vehicle detected at
each point, decide which A-vehicle corresponds to which B-vehicle. Two signals
are combined:

    * **Appearance** — cosine similarity between L2-normalized embeddings.
      Because vectors are unit-length, cosine similarity is just a dot product.
    * **Time** — a vehicle must pass A *before* B, within a plausible travel
      window. Pairs outside the window are removed before ranking.

The core numeric routines take plain numpy arrays and are dependency-light so
they unit-test without any model or network.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

import config


@dataclass
class VehicleRecord:
    """A single detected vehicle plus everything needed to match and display it."""

    record_id: int          # unique index within its point's list
    point: str              # "A" or "B"
    frame_path: str
    timestamp: datetime
    bbox: tuple[int, int, int, int]
    confidence: float
    embedding: np.ndarray   # (dim,) L2-normalized


@dataclass
class MatchCandidate:
    """One ranked B-candidate for a given A-vehicle."""

    b_record_id: int
    similarity: float


@dataclass
class MatchResult:
    """All accepted candidates for a single A-vehicle, best first."""

    a_record_id: int
    candidates: list[MatchCandidate]

    @property
    def best(self) -> MatchCandidate | None:
        return self.candidates[0] if self.candidates else None


def cosine_similarity_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Return the (len(A), len(B)) cosine-similarity matrix.

    Inputs are assumed L2-normalized (the embedder guarantees this), so the
    similarity is a plain matrix product. Empty inputs yield a correctly-shaped
    empty array.
    """
    if emb_a.size == 0 or emb_b.size == 0:
        return np.zeros((len(emb_a), len(emb_b)), dtype=np.float32)
    return emb_a.astype(np.float32) @ emb_b.astype(np.float32).T


def time_gate_mask(
    times_a: list[datetime],
    times_b: list[datetime],
    min_travel_seconds: float,
    max_travel_seconds: float,
) -> np.ndarray:
    """Boolean (len(A), len(B)) mask of pairs whose travel time is plausible.

    ``mask[i, j]`` is True when ``min <= (t_b[j] - t_a[i]) <= max`` seconds.
    A vehicle passing B before A (negative delta) is naturally excluded when
    ``min_travel_seconds`` is >= 0.
    """
    na, nb = len(times_a), len(times_b)
    mask = np.zeros((na, nb), dtype=bool)
    for i, ta in enumerate(times_a):
        for j, tb in enumerate(times_b):
            delta = (tb - ta).total_seconds()
            mask[i, j] = min_travel_seconds <= delta <= max_travel_seconds
    return mask


def match(
    records_a: list[VehicleRecord],
    records_b: list[VehicleRecord],
    cfg: config.MatchConfig | None = None,
) -> list[MatchResult]:
    """Match A-vehicles to B-vehicles under appearance + temporal constraints.

    Returns one ``MatchResult`` per A-vehicle (in input order), each holding up
    to ``cfg.top_k`` accepted candidates ranked by similarity. When
    ``cfg.one_to_one`` is set, a global one-to-one assignment (Hungarian) is
    computed first and each A keeps at most that single partner.
    """
    cfg = cfg or config.MatchConfig()

    emb_a = np.stack([r.embedding for r in records_a]) if records_a else np.zeros((0, 1))
    emb_b = np.stack([r.embedding for r in records_b]) if records_b else np.zeros((0, 1))

    sim = cosine_similarity_matrix(emb_a, emb_b)

    # Apply the temporal gate by driving disallowed pairs to -inf so they can
    # never be selected or pass the threshold.
    if cfg.use_time_gate and sim.size:
        mask = time_gate_mask(
            [r.timestamp for r in records_a],
            [r.timestamp for r in records_b],
            cfg.min_travel_seconds,
            cfg.max_travel_seconds,
        )
        sim = np.where(mask, sim, -np.inf)

    if cfg.one_to_one and sim.size and np.isfinite(sim).any():
        return _match_one_to_one(records_a, records_b, sim, cfg)

    return _match_top_k(records_a, records_b, sim, cfg)


def _match_top_k(
    records_a: list[VehicleRecord],
    records_b: list[VehicleRecord],
    sim: np.ndarray,
    cfg: config.MatchConfig,
) -> list[MatchResult]:
    results: list[MatchResult] = []
    for i, rec_a in enumerate(records_a):
        candidates: list[MatchCandidate] = []
        if sim.shape[1] > 0:
            order = np.argsort(-sim[i])  # descending similarity
            for j in order[: cfg.top_k]:
                score = float(sim[i, j])
                if not np.isfinite(score) or score < cfg.similarity_threshold:
                    continue
                candidates.append(
                    MatchCandidate(b_record_id=records_b[j].record_id, similarity=score)
                )
        results.append(MatchResult(a_record_id=rec_a.record_id, candidates=candidates))
    return results


def cluster_same_point(
    records: list[VehicleRecord],
    similarity_threshold: float = config.DEFAULT_SAME_POINT_SIMILARITY_THRESHOLD,
) -> dict[int, int]:
    """Group records from ONE point that likely show the same physical vehicle.

    The same vehicle can be detected in several frames at a single point (e.g.
    waiting, circling back). This groups such detections by appearance only —
    no time gate, since all detections are already known to be at the same
    point, so recording order doesn't constrain which ones can match.

    Returns ``{record_id: cluster_id}`` with every record assigned a cluster,
    including singletons (cluster ids are 0-indexed in first-seen order).
    """
    n = len(records)
    if n == 0:
        return {}

    emb = np.stack([r.embedding for r in records])
    sim = cosine_similarity_matrix(emb, emb)

    # Union-find over indices connected by similarity >= threshold.
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

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= similarity_threshold:
                union(i, j)

    root_to_cluster: dict[int, int] = {}
    result: dict[int, int] = {}
    for i, rec in enumerate(records):
        root = find(i)
        if root not in root_to_cluster:
            root_to_cluster[root] = len(root_to_cluster)
        result[rec.record_id] = root_to_cluster[root]
    return result


def _match_one_to_one(
    records_a: list[VehicleRecord],
    records_b: list[VehicleRecord],
    sim: np.ndarray,
    cfg: config.MatchConfig,
) -> list[MatchResult]:
    from scipy.optimize import linear_sum_assignment

    # Hungarian minimizes cost; use a large finite cost for gated-out pairs so
    # the solver still returns a complete assignment.
    big = 1e6
    cost = np.where(np.isfinite(sim), -sim, big)
    row_ind, col_ind = linear_sum_assignment(cost)
    assigned: dict[int, int] = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    results: list[MatchResult] = []
    for i, rec_a in enumerate(records_a):
        candidates: list[MatchCandidate] = []
        j = assigned.get(i)
        if j is not None and j < sim.shape[1]:
            score = float(sim[i, j])
            if np.isfinite(score) and score >= cfg.similarity_threshold:
                candidates.append(
                    MatchCandidate(b_record_id=records_b[j].record_id, similarity=score)
                )
        results.append(MatchResult(a_record_id=rec_a.record_id, candidates=candidates))
    return results
