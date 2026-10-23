#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

RESOLUTION = 5000
Scored = Tuple[str, int, int, float]
TAD = Tuple[str, int, int]


def to_bins(left_bp: int, right_bp: int) -> Tuple[int, int]:
    return left_bp // RESOLUTION, right_bp // RESOLUTION


def _conflicts(sel_set: Set[Tuple[str, int, int]], chrom: str, l_bin: int, r_bin: int, gap_bins: int) -> bool:
    for dl in range(-gap_bins, gap_bins + 1):
        for dr in range(-gap_bins, gap_bins + 1):
            if (chrom, l_bin + dl, r_bin + dr) in sel_set:
                return True
    return False


def greedy_dedup(candidates: List[Scored], dedup_gap_bins: int = 5) -> List[TAD]:
    selected: List[TAD] = []
    sel_set: Set[Tuple[str, int, int]] = set()
    for chrom, left, right, _score in sorted(candidates, key=lambda x: x[3], reverse=True):
        l_bin, r_bin = to_bins(left, right)
        if l_bin >= r_bin:
            continue
        if _conflicts(sel_set, chrom, l_bin, r_bin, dedup_gap_bins):
            continue
        selected.append((chrom, left, right))
        sel_set.add((chrom, l_bin, r_bin))
    return selected


def _build_snap_map(boundary_bins: List[int], snap_bin: int) -> Dict[int, int]:
    if not boundary_bins or snap_bin <= 0:
        return {b: b for b in set(boundary_bins)}
    counts = Counter(boundary_bins)
    uniq = sorted(counts.keys())
    clusters: List[List[int]] = []
    cur = [uniq[0]]
    for b in uniq[1:]:
        if b - cur[-1] <= snap_bin:
            cur.append(b)
        else:
            clusters.append(cur)
            cur = [b]
    clusters.append(cur)
    mapping: Dict[int, int] = {}
    for cl in clusters:
        total = sum(counts[v] for v in cl)
        center = sum(v * counts[v] for v in cl) / max(total, 1)
        anchor = min(cl, key=lambda v: (abs(v - center), -counts[v]))
        for v in cl:
            mapping[v] = anchor
    return mapping


def snap_boundaries(tads: List[TAD], snap_bin: int) -> Tuple[List[TAD], dict]:
    if snap_bin <= 0 or not tads:
        return list(tads), {
            "snap_bin": snap_bin,
            "total_before": len(tads),
            "total_after": len(tads),
            "unique_left_before": len(set((c, s // RESOLUTION) for c, s, _ in tads)),
            "unique_right_before": len(set((c, e // RESOLUTION) for c, _, e in tads)),
            "unique_left_after": len(set((c, s // RESOLUTION) for c, s, _ in tads)),
            "unique_right_after": len(set((c, e // RESOLUTION) for c, _, e in tads)),
        }
    by_chrom: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for chrom, left, right in tads:
        l_bin, r_bin = to_bins(left, right)
        if r_bin > l_bin:
            by_chrom[chrom].append((l_bin, r_bin))
    snapped_set: Set[TAD] = set()
    unique_left_before = unique_right_before = unique_left_after = unique_right_after = 0
    for chrom, pairs in by_chrom.items():
        lefts = [l for l, _ in pairs]
        rights = [r for _, r in pairs]
        unique_left_before += len(set(lefts))
        unique_right_before += len(set(rights))
        l_map = _build_snap_map(lefts, snap_bin)
        r_map = _build_snap_map(rights, snap_bin)
        for l, r in pairs:
            sl = l_map.get(l, l)
            sr = r_map.get(r, r)
            if sr <= sl:
                sl, sr = l, r
            if sr > sl:
                snapped_set.add((chrom, sl * RESOLUTION, sr * RESOLUTION))
        unique_left_after += len(set(l_map.values()))
        unique_right_after += len(set(r_map.values()))
    snapped = sorted(snapped_set, key=lambda x: (x[0], x[1], x[2]))
    return snapped, {
        "snap_bin": snap_bin,
        "total_before": len(tads),
        "total_after": len(snapped),
        "unique_left_before": unique_left_before,
        "unique_right_before": unique_right_before,
        "unique_left_after": unique_left_after,
        "unique_right_after": unique_right_after,
    }
