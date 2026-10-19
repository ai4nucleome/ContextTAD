"""Final-TAD target/decoy calibration for ContextTAD boundary tracks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import post_process
import predict_structure


RESOLUTION = predict_structure.RESOLUTION


def shift_distance_bands(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Build one symmetric decoy map by cyclically shifting distance bands."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square contact map")
    n = len(matrix)
    output = np.empty_like(matrix)
    shifts: dict[tuple[str, int], int] = {}
    for distance in range(n):
        band = ("near", distance // 3) if distance < 20 else ("far", (distance - 20) // 5)
        if band not in shifts:
            shift = int(rng.integers(max(10, n // 8), max(11, n // 2)))
            shifts[band] = shift if rng.random() < .5 else -shift
        values = np.diag(matrix, k=distance).copy()
        if len(values) > 1:
            values = np.roll(values, shifts[band] % len(values))
        positions = np.arange(n - distance)
        output[positions, positions + distance] = values
        output[positions + distance, positions] = values
    return output


def assemble(chrom: str, left: np.ndarray, right: np.ndarray, peak_radius: int = 4):
    lp, _ = predict_structure.peak_metrics(left, peak_radius, "prominence")
    rp, _ = predict_structure.peak_metrics(right, peak_radius, "prominence")
    pairs = predict_structure.pair_boundaries(
        left, right, lp, rp, predict_structure.MIN_LEN_BINS,
        predict_structure.MAX_LEN_BINS, predict_structure.NMS_METRIC,
        predict_structure.NMS_THRESH, predict_structure.ALLOW_NESTED,
    )
    return [(chrom, l * RESOLUTION, (r + 1) * RESOLUTION, float(score))
            for l, r, score in pairs]


def prune(pairs):
    """Keep nested parents supported by a top-five pair score, as in v32."""
    if not pairs:
        return []
    left = np.asarray([p[1] for p in pairs], dtype=np.int64)
    right = np.asarray([p[2] for p in pairs], dtype=np.int64)
    scores = np.asarray([p[3] for p in pairs], dtype=np.float64)
    lengths = right - left
    contains = ((left[:, None] <= left[None, :]) &
                (right[:, None] >= right[None, :]))
    np.fill_diagonal(contains, False)
    child_counts = contains.sum(axis=1)
    child_lengths = contains @ lengths
    keep = np.ones(len(pairs), dtype=bool)
    for i in np.flatnonzero(child_counts):
        passes = child_counts[i] <= 5
        if not passes:
            fifth = np.partition(scores[contains[i]], -5)[-5]
            passes = scores[i] >= fifth
        keep[i] = (passes and lengths[i] <= 2_000_000 and
                   child_lengths[i] <= 3 * lengths[i])
    return [pair for pair, selected in zip(pairs, keep) if selected]


def final_calls(pairs):
    snapped, _ = post_process.snap_boundaries(
        [(chrom, start, end) for chrom, start, end, _ in pairs], snap_bin=2)
    return {(chrom, start, end - RESOLUTION)
            for chrom, start, end in snapped if end - RESOLUTION > start}


@dataclass
class Prefix:
    pairs: list[tuple[str, int, int, float]]
    ascending_scores: np.ndarray
    survivor_counts: np.ndarray

    def counts_at(self, cutoffs: np.ndarray) -> np.ndarray:
        n = len(self.pairs) - np.searchsorted(self.ascending_scores, cutoffs, side="left")
        return self.survivor_counts[n]

    def calls_at(self, cutoff: float):
        return final_calls(prune([pair for pair in self.pairs if pair[3] >= cutoff]))


def prepare(raw) -> Prefix:
    if not raw:
        return Prefix([], np.zeros(0), np.zeros(1, dtype=np.int32))
    best_score = {}
    for chrom, left, right, score in raw:
        key = chrom, left, right
        best_score[key] = max(score, best_score.get(key, float("-inf")))
    dedup = post_process.greedy_dedup(raw, dedup_gap_bins=5)
    pairs = [(chrom, left, right, best_score[(chrom, left, right)])
             for chrom, left, right in dedup]
    n = len(pairs)
    scores = np.asarray([p[3] for p in pairs], dtype=np.float64)
    if np.any(scores[:-1] < scores[1:]):
        raise AssertionError("Deduplicated candidates must be score ordered")
    left = np.asarray([p[1] for p in pairs], dtype=np.int64)
    right = np.asarray([p[2] for p in pairs], dtype=np.int64)
    lengths = right - left
    children = np.zeros(n, dtype=np.int32)
    child_lengths = np.zeros(n, dtype=np.int64)
    rescue = np.zeros(n, dtype=bool)
    kept = np.zeros(n, dtype=bool)
    counts = np.zeros(n + 1, dtype=np.int32)
    total = 0
    for i in range(n):
        prior_parent = (left[:i] <= left[i]) & (right[:i] >= right[i])
        affected = np.flatnonzero(prior_parent)
        old_kept = int(kept[affected].sum())
        children[affected] += 1
        child_lengths[affected] += lengths[i]
        kept[affected] = (((children[affected] <= 5) | rescue[affected])
                          & (lengths[affected] <= 2_000_000)
                          & (child_lengths[affected] <= 3 * lengths[affected]))
        total += int(kept[affected].sum()) - old_kept
        prior_child = (left[:i] >= left[i]) & (right[:i] <= right[i])
        children[i] = int(prior_child.sum())
        child_lengths[i] = int(lengths[:i][prior_child].sum())
        rescue[i] = np.count_nonzero(scores[:i][prior_child] > scores[i]) < 5
        kept[i] = (children[i] == 0 or
                   ((children[i] <= 5 or rescue[i])
                    and lengths[i] <= 2_000_000
                    and child_lengths[i] <= 3 * lengths[i]))
        total += int(kept[i])
        counts[i + 1] = total
    return Prefix(pairs, scores[::-1], counts)


def calibrate(target: Prefix, decoy: Prefix, alpha: float):
    if not target.pairs:
        return float("inf"), set(), set()
    cutoffs = np.unique(np.concatenate((target.ascending_scores, decoy.ascending_scores)))
    nt = target.counts_at(cutoffs)
    nd = decoy.counts_at(cutoffs)
    valid = np.flatnonzero((nt > 0) & (nd <= alpha * nt))
    ranked = sorted(valid, key=lambda i: (nt[i], -nd[i] / nt[i], cutoffs[i]), reverse=True)
    for i in ranked:
        cutoff = float(cutoffs[i])
        target_calls = target.calls_at(cutoff)
        decoy_calls = decoy.calls_at(cutoff)
        if target_calls and len(decoy_calls) <= alpha * len(target_calls):
            return cutoff, target_calls, decoy_calls
    return float("inf"), set(), set()
