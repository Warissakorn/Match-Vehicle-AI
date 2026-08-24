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
    # Where ``timestamp`` came from: "timeline" | "ocr" | "filename" |
    # "exif" | "mtime".
    # Defaulted so records pickled before this field existed still unpickle.
    timestamp_source: str = "unknown"


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

    Broadcast rather than looped: this used to be a Python double loop, which
    at a real gallery size (3,670 x 12,557) is 46 million iterations and took
    ~21 s. The GUI never noticed, because it matches one A-vehicle at a time,
    but the CLI passes all of A at once. Measured 10x faster with identical
    output.

    Seconds are measured relative to the first A timestamp, not the Unix
    epoch: float32 near 1.7e9 has a resolution of about 128 s, which would
    wreck the comparison, while relative offsets stay well under a day where
    float32 resolves to a few milliseconds -- far finer than any travel-time
    threshold. That keeps the intermediate array half the size of float64,
    which matters because it is the largest allocation in a match.
    """
    na, nb = len(times_a), len(times_b)
    if na == 0 or nb == 0:
        return np.zeros((na, nb), dtype=bool)

    origin = times_a[0]
    seconds_a = np.fromiter(((t - origin).total_seconds() for t in times_a),
                            dtype=np.float32, count=na)
    seconds_b = np.fromiter(((t - origin).total_seconds() for t in times_b),
                            dtype=np.float32, count=nb)
    delta = seconds_b[None, :] - seconds_a[:, None]
    return (delta >= min_travel_seconds) & (delta <= max_travel_seconds)


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

    The similarity matrix is materialised **one row-block at a time** rather
    than all at once. At a real gallery size (3,670 x 12,557) the old dense
    path held the float32 similarities *and* the boolean gate mask *and* the
    ``np.where`` result simultaneously -- three full-size allocations for data
    only ever consumed a row at a time. Blocking keeps peak memory at one
    block (~20 MB at 1024 rows) regardless of gallery size, with identical
    output; see ``_gated_block``.
    """
    cfg = cfg or config.MatchConfig()

    if not records_a:
        return []
    if not records_b:
        return [MatchResult(a_record_id=r.record_id, candidates=[]) for r in records_a]

    emb_a = np.stack([r.embedding for r in records_a]).astype(np.float32)
    emb_b = np.stack([r.embedding for r in records_b]).astype(np.float32)

    times_a = [r.timestamp for r in records_a]
    times_b = [r.timestamp for r in records_b]
    sec_b = _relative_seconds(times_b, times_a[0])

    if cfg.one_to_one:
        cost, any_finite = _build_cost_matrix(
            emb_a, emb_b, times_a, sec_b, cfg)
        if any_finite:
            return _match_one_to_one(records_a, records_b, cost, cfg)

    return _match_top_k_streaming(emb_a, emb_b, times_a, sec_b, records_a,
                                  records_b, cfg)


#: Rows of A processed per streaming block. Bounds the block's memory at
#: ``_ROW_BLOCK x len(B) x 4`` bytes -- ~50 MB against a 12k-record B gallery.
_ROW_BLOCK = 1024


def _relative_seconds(times: list[datetime], origin: datetime) -> np.ndarray:
    """Timestamps as float32 seconds relative to ``origin``.

    Float32 near the Unix epoch resolves to ~128 s -- useless for travel-time
    comparisons -- while offsets from a common origin stay well under a day,
    where float32 resolves to milliseconds. See ``time_gate_mask``.
    """
    return np.fromiter(((t - origin).total_seconds() for t in times),
                       dtype=np.float32, count=len(times))


def _gated_block(
    emb_a_block: np.ndarray,
    sec_a_block: np.ndarray,
    emb_b: np.ndarray,
    sec_b: np.ndarray,
    use_time_gate: bool,
    min_travel_seconds: float,
    max_travel_seconds: float,
) -> np.ndarray:
    """Similarities for one block of A rows, gated pairs driven to ``-inf``.

    This is the per-block equivalent of ``cosine_similarity_matrix`` followed
    by ``np.where(mask, sim, -inf)``, computed against a slice of A so no
    full-gallery temporary ever exists. Gated-out values become ``-inf``
    (not merely masked) so neither ranking nor the threshold test can pick
    them, exactly as before.
    """
    sim = emb_a_block @ emb_b.T
    if use_time_gate:
        delta = sec_b[None, :] - sec_a_block[:, None]
        sim[(delta < min_travel_seconds) | (delta > max_travel_seconds)] = -np.inf
    return sim


def _blocks(n: int):
    """Yield ``(start, stop)`` row ranges of at most ``_ROW_BLOCK`` rows."""
    for start in range(0, n, _ROW_BLOCK):
        yield start, min(start + _ROW_BLOCK, n)


def _match_top_k_streaming(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    times_a: list[datetime],
    sec_b: np.ndarray,
    records_a: list[VehicleRecord],
    records_b: list[VehicleRecord],
    cfg: config.MatchConfig,
) -> list[MatchResult]:
    results: list[MatchResult] = []
    origin = times_a[0]
    for start, stop in _blocks(len(records_a)):
        sec_blk = _relative_seconds(times_a[start:stop], origin)
        sim_blk = _gated_block(emb_a[start:stop], sec_blk, emb_b, sec_b,
                               cfg.use_time_gate, cfg.min_travel_seconds,
                               cfg.max_travel_seconds)
        for offset, rec_a in enumerate(records_a[start:stop]):
            row = sim_blk[offset]
            candidates: list[MatchCandidate] = []
            order = np.argsort(-row)  # descending similarity
            for j in order[: cfg.top_k]:
                score = float(row[j])
                if not np.isfinite(score) or score < cfg.similarity_threshold:
                    continue
                candidates.append(
                    MatchCandidate(b_record_id=records_b[j].record_id, similarity=score)
                )
            results.append(MatchResult(a_record_id=rec_a.record_id, candidates=candidates))
    return results


def _build_cost_matrix(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    times_a: list[datetime],
    sec_b: np.ndarray,
    cfg: config.MatchConfig,
) -> tuple[np.ndarray, bool]:
    """Hungarian cost matrix built one block at a time.

    ``linear_sum_assignment`` needs the whole matrix up front -- that part is
    unavoidable -- but building it blockwise folds the gate mask into the
    fill instead of allocating separate mask/``where`` temporaries beside it.
    Returns ``(cost, any_finite)``; an all-gated problem has no meaningful
    assignment and the caller falls back to top-k, matching the old
    ``np.isfinite(sim).any()`` guard.
    """
    big = 1e6
    cost = np.empty((len(emb_a), len(emb_b)), dtype=np.float32)
    any_finite = False
    origin = times_a[0]
    for start, stop in _blocks(len(emb_a)):
        sec_blk = _relative_seconds(times_a[start:stop], origin)
        blk = _gated_block(emb_a[start:stop], sec_blk, emb_b, sec_b,
                           cfg.use_time_gate, cfg.min_travel_seconds,
                           cfg.max_travel_seconds)
        finite = np.isfinite(blk)
        any_finite |= bool(finite.any())
        blk[~finite] = big
        np.negative(blk, out=blk, where=finite)
        cost[start:stop] = blk
    return cost, any_finite


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

    emb = np.stack([r.embedding for r in records]).astype(np.float32)

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

    # Find qualifying (i, j) pairs with numpy rather than a Python-level n^2
    # loop: with thousands of vehicles per point (real gallery sizes seen in
    # the field run into the tens of thousands) a pure-Python double loop is
    # tens of millions of iterations and freezes the caller for minutes.
    # The similarity pass runs one row-block at a time (as in ``match``), and
    # each block keeps only its strict-upper-triangle pairs above threshold --
    # since genuine repeat sightings are a small fraction of all pairs, only
    # the handful of qualifying indices ever reach the Python union-find loop.
    # Union-find is order-independent, so visiting pairs block-by-block gives
    # exactly the clusters the old full-matrix np.triu walk produced.
    cols = np.arange(n)
    for start, stop in _blocks(n):
        blk = emb[start:stop] @ emb.T
        upper = cols[None, :] > np.arange(start, stop)[:, None]
        pairs_local, pairs_j = np.nonzero((blk >= similarity_threshold) & upper)
        for offset, j in zip(pairs_local.tolist(), pairs_j.tolist()):
            union(start + offset, int(j))

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
    cost: np.ndarray,
    cfg: config.MatchConfig,
) -> list[MatchResult]:
    """Hungarian assignment over the prebuilt cost matrix (see
    ``_build_cost_matrix``).

    The solver always returns a complete assignment -- gated-out pairs carry
    the large sentinel cost ``1e6``, far outside cosine similarity's
    ``[-1, 1]`` -- so an assigned pair is only reported when its cost is a
    real similarity at or above the threshold, which is exactly what the old
    ``np.isfinite(sim[i, j])`` test said.
    """
    from scipy.optimize import linear_sum_assignment

    big = 1e6
    row_ind, col_ind = linear_sum_assignment(cost)
    assigned: dict[int, int] = {int(r): int(c) for r, c in zip(row_ind, col_ind)}

    results: list[MatchResult] = []
    for i, rec_a in enumerate(records_a):
        candidates: list[MatchCandidate] = []
        j = assigned.get(i)
        if j is not None and j < cost.shape[1]:
            c = float(cost[i, j])
            if c < big and -c >= cfg.similarity_threshold:
                candidates.append(
                    MatchCandidate(b_record_id=records_b[j].record_id, similarity=-c)
                )
        results.append(MatchResult(a_record_id=rec_a.record_id, candidates=candidates))
    return results
