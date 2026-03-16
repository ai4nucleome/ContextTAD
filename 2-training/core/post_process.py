#!/usr/bin/env python3
"""Minimal keep-ratio postprocess for ContextTAD raw BED.

Pipeline:
  1) length filter
  2) keep top ratio by score
  3) near-duplicate removal in bin space
  4) prune over-complex L1+ structures
  5) export 3-col BED in right-boundary coordinates (right = end_exclusive - 1 bin)
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

RESOLUTION = 5000


def read_scored_bed(path: Path, min_len_bins: int = 5) -> List[Tuple[str, int, int, float]]:
    items: List[Tuple[str, int, int, float]] = []
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                chrom = parts[0]
                left = int(parts[1])
                right = int(parts[2])
                score = float(parts[3]) if len(parts) >= 4 else 0.5
            except Exception:
                continue
            if right <= left:
                continue

            span = right - left
            if left % RESOLUTION == 0 and span % RESOLUTION == 0:
                length_bins = span // RESOLUTION
            else:
                length_bins = span
            if length_bins < min_len_bins:
                continue

            items.append((chrom, left, right, score))
    return items


def to_bins(left_bp: int, right_bp: int) -> Tuple[int, int]:
    return left_bp // RESOLUTION, right_bp // RESOLUTION


def _conflicts(sel_set: Set[Tuple[str, int, int]], chrom: str, l_bin: int, r_bin: int, gap_bins: int) -> bool:
    for dl in range(-gap_bins, gap_bins + 1):
        for dr in range(-gap_bins, gap_bins + 1):
            if (chrom, l_bin + dl, r_bin + dr) in sel_set:
                return True
    return False


def greedy_dedup(candidates: List[Tuple[str, int, int, float]], dedup_gap_bins: int = 4) -> List[Tuple[str, int, int]]:
    sorted_cands = sorted(candidates, key=lambda x: x[3], reverse=True)
    selected: List[Tuple[str, int, int]] = []
    sel_set: Set[Tuple[str, int, int]] = set()

    for chrom, left, right, _score in sorted_cands:
        l_bin, r_bin = to_bins(left, right)
        if l_bin >= r_bin:
            continue
        if _conflicts(sel_set, chrom, l_bin, r_bin, dedup_gap_bins):
            continue
        selected.append((chrom, left, right))
        sel_set.add((chrom, l_bin, r_bin))

    return selected


def classify_l0_l1plus(tads: List[Tuple[str, int, int]]):
    by_chrom: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for chrom, start, end in tads:
        by_chrom[chrom].append((start, end))

    l0 = []
    l1plus = []
    children_map = {}

    for chrom, chrom_tads in by_chrom.items():
        for tad in chrom_tads:
            children = []
            for tad2 in chrom_tads:
                if tad2[0] >= tad[0] and tad2[1] <= tad[1]:
                    if tad2[0] == tad[0] and tad2[1] == tad[1]:
                        continue
                    children.append(tad2)
            key = (chrom, tad[0], tad[1])
            children_map[key] = children
            if len(children) > 0:
                l1plus.append(key)
            else:
                l0.append(key)

    return l0, l1plus, children_map


def prune_l1plus(
    tads: List[Tuple[str, int, int]],
    max_children: int,
    max_l1plus_size: int,
    max_children_coverage_ratio: float,
) -> Tuple[List[Tuple[str, int, int]], dict]:
    l0, l1plus, children_map = classify_l0_l1plus(tads)

    stats = {
        "total_before": len(tads),
        "l0_count": len(l0),
        "l1plus_before": len(l1plus),
        "dropped_children": 0,
        "dropped_size": 0,
        "dropped_coverage_ratio": 0,
        "l1plus_kept": 0,
    }

    kept_l1plus = []
    for tad_key in l1plus:
        _chrom, start, end = tad_key
        children = children_map[tad_key]
        n_children = len(children)
        tad_size = end - start
        child_total_size = sum(c[1] - c[0] for c in children)
        coverage_ratio = child_total_size / max(tad_size, 1)

        if n_children > max_children:
            stats["dropped_children"] += 1
            continue
        if tad_size > max_l1plus_size:
            stats["dropped_size"] += 1
            continue
        if coverage_ratio > max_children_coverage_ratio:
            stats["dropped_coverage_ratio"] += 1
            continue

        kept_l1plus.append(tad_key)

    stats["l1plus_kept"] = len(kept_l1plus)
    stats["l1plus_dropped"] = stats["l1plus_before"] - stats["l1plus_kept"]
    survivors = l0 + kept_l1plus
    stats["total_after"] = len(survivors)

    return survivors, stats


def keep_top_ratio(candidates: List[Tuple[str, int, int, float]], keep_ratio: float):
    keep_ratio = max(min(float(keep_ratio), 1.0), 0.0)
    if keep_ratio <= 0.0:
        return [], None
    if keep_ratio >= 1.0:
        return list(candidates), None

    ranked = sorted(candidates, key=lambda x: x[3], reverse=True)
    k = max(1, int(round(len(ranked) * keep_ratio)))
    kept = ranked[:k]
    eff_thr = kept[-1][3]
    return kept, eff_thr


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


def snap_boundaries(tads: List[Tuple[str, int, int]], snap_bin: int) -> Tuple[List[Tuple[str, int, int]], dict]:
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
        if r_bin <= l_bin:
            continue
        by_chrom[chrom].append((l_bin, r_bin))

    snapped_set: Set[Tuple[str, int, int]] = set()
    unique_left_before = 0
    unique_right_before = 0
    unique_left_after = 0
    unique_right_after = 0

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
            if sr <= sl:
                continue
            snapped_set.add((chrom, sl * RESOLUTION, sr * RESOLUTION))

        unique_left_after += len(set(l_map.values()))
        unique_right_after += len(set(r_map.values()))

    snapped = sorted(snapped_set, key=lambda x: (x[0], x[1], x[2]))
    stats = {
        "snap_bin": snap_bin,
        "total_before": len(tads),
        "total_after": len(snapped),
        "unique_left_before": unique_left_before,
        "unique_right_before": unique_right_before,
        "unique_left_after": unique_left_after,
        "unique_right_after": unique_right_after,
    }
    return snapped, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Keep-ratio ContextTAD postprocess")
    ap.add_argument("--raw_bed", required=True, help="Input 4-column BED")
    ap.add_argument("--output_bed", required=True, help="Output 3-column BED")
    ap.add_argument("--stats_json", default="", help="Optional JSON stats output")
    ap.add_argument("--min_len_bins", type=int, default=5)
    ap.add_argument("--keep_ratio", type=float, default=0.66, help="Keep top ratio by score before dedup")
    ap.add_argument("--dedup_gap_bins", type=int, default=4)
    ap.add_argument("--max_children", type=int, default=12)
    ap.add_argument("--max_l1plus_size", type=int, default=2_000_000)
    ap.add_argument("--max_children_coverage_ratio", type=float, default=2.5)
    ap.add_argument("--snap_bin", type=int, default=2, help="Boundary snapping radius in bins")
    args = ap.parse_args()

    raw_path = Path(args.raw_bed)
    if not raw_path.is_file():
        print(f"ERROR: raw_bed not found: {raw_path}", file=sys.stderr)
        sys.exit(1)

    candidates = read_scored_bed(raw_path, min_len_bins=args.min_len_bins)
    print(f"Loaded {len(candidates)} candidates from {raw_path}")

    kept, eff_thr = keep_top_ratio(candidates, args.keep_ratio)
    print(
        f"Top-ratio keep ({args.keep_ratio:.4f}): {len(candidates)} -> {len(kept)} "
        f"(effective_score_threshold={eff_thr})"
    )

    out_path = Path(args.output_bed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not kept:
        out_path.write_text("")
        print("No candidates kept; wrote empty BED.")
        return

    deduped = greedy_dedup(kept, dedup_gap_bins=args.dedup_gap_bins)
    print(f"After dedup (gap={args.dedup_gap_bins}): {len(deduped)} TADs")

    pruned, stats = prune_l1plus(
        deduped,
        max_children=args.max_children,
        max_l1plus_size=args.max_l1plus_size,
        max_children_coverage_ratio=args.max_children_coverage_ratio,
    )
    print(f"After L1+ pruning: {stats['total_before']} -> {stats['total_after']}")

    snapped, snap_stats = snap_boundaries(pruned, snap_bin=args.snap_bin)
    print(
        "After snapping: "
        f"count {snap_stats['total_before']} -> {snap_stats['total_after']}; "
        f"left uniq {snap_stats['unique_left_before']} -> {snap_stats['unique_left_after']}; "
        f"right uniq {snap_stats['unique_right_before']} -> {snap_stats['unique_right_after']}"
    )

    with out_path.open("w") as f:
        n_write = 0
        for chrom, left, right in snapped:
            # Final output uses right-boundary coordinate.
            right_out = right - RESOLUTION
            if right_out <= left:
                continue
            f.write(f"{chrom}\t{left}\t{right_out}\n")
            n_write += 1

    print(f"Written {n_write} TADs to {out_path}")

    if args.stats_json:
        stats_obj = {
            "raw_bed": str(raw_path),
            "output_bed": str(out_path),
            "n_candidates": len(candidates),
            "n_kept_ratio": len(kept),
            "effective_score_threshold": eff_thr,
            "n_dedup": len(deduped),
            "prune": stats,
            "snap": snap_stats,
            "n_written": n_write,
            "params": {
                "min_len_bins": args.min_len_bins,
                "keep_ratio": args.keep_ratio,
                "dedup_gap_bins": args.dedup_gap_bins,
                "max_children": args.max_children,
                "max_l1plus_size": args.max_l1plus_size,
                "max_children_coverage_ratio": args.max_children_coverage_ratio,
                "snap_bin": args.snap_bin,
            },
        }
        stats_path = Path(args.stats_json)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats_obj, indent=2) + "\n")
        print(f"Stats written to {stats_path}")


if __name__ == "__main__":
    main()
