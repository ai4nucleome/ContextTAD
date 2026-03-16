#!/usr/bin/env python3
"""
Build whole-genome ground-truth BED from all window splits.

This script collects train/val/test windows, merges them into chromosome-level
intervals, and deduplicates endpoints using a tolerance in base pairs.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get("TAD_DATA_DIR", "/home/weicai/projectnvme/TADAnno_final/0-data/1_dp_train_infer_data"))
RESOLUTION = 5000
AUTOSOME_CHROMS = [f"chr{i}" for i in range(1, 23)]


def _parse_window_name(window_name):
    chrom, offset = window_name.rsplit("_", 1)
    return chrom, int(offset)


def _chrom_sort_key(chrom):
    if chrom.startswith("chr"):
        suffix = chrom[3:]
        if suffix.isdigit():
            return int(suffix)
    return 10**9


def _collect_all_windows(window_list_json):
    windows = []
    win_obj = window_list_json.get("windows", {})
    if isinstance(win_obj, dict):
        for split_name in ("train", "val", "test"):
            windows.extend(win_obj.get(split_name, []))
    elif isinstance(win_obj, list):
        windows.extend(win_obj)

    seen = set()
    uniq = []
    for wn in windows:
        if wn in seen:
            continue
        seen.add(wn)
        uniq.append(wn)
    return uniq


def build_gt_all(data_dir, output_path, tolerance_bp=5000, target_chroms=None):
    if target_chroms is None:
        target_chroms = AUTOSOME_CHROMS

    with open(data_dir / "window_list.json") as f:
        wl = json.load(f)

    all_windows = _collect_all_windows(wl)

    raw_tads = defaultdict(list)  # {chrom: [(start_bp, end_bp), ...]}
    window_count = defaultdict(int)

    for wn in all_windows:
        try:
            chrom, offset = _parse_window_name(wn)
        except ValueError:
            continue

        if chrom not in target_chroms:
            continue

        tads_path = data_dir / "labels" / chrom / f"{wn}_tads.npy"
        if not tads_path.exists():
            continue

        window_count[chrom] += 1
        tads = np.array(np.load(tads_path))
        if tads.size == 0:
            continue
        if tads.ndim == 1:
            if tads.shape[0] < 2:
                continue
            tads = tads.reshape(1, -1)

        for row in tads.tolist():
            if len(row) < 2:
                continue
            left = int(row[0])
            right = int(row[1])
            if left < 0:
                continue
            start_bp = (offset + int(left)) * RESOLUTION
            end_bp = (offset + int(right) + 1) * RESOLUTION
            raw_tads[chrom].append((start_bp, end_bp))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(output_path, "w") as f:
        for chrom in sorted(raw_tads, key=_chrom_sort_key):
            sorted_tads = sorted(set(raw_tads[chrom]))
            deduped = []

            for s, e in sorted_tads:
                is_dup = False
                for ds, de in deduped:
                    if abs(s - ds) <= tolerance_bp and abs(e - de) <= tolerance_bp:
                        is_dup = True
                        break
                if not is_dup:
                    deduped.append((s, e))

            for s, e in deduped:
                f.write(f"{chrom}\t{s}\t{e}\n")
                total += 1

            print(
                f"  {chrom}: {window_count[chrom]} windows, "
                f"{len(raw_tads[chrom])} raw  {len(deduped)} deduped"
            )

    print(f"Total: {total} TADs  {output_path}")
    return total


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "TAD_dp_all.bed"),
    )
    p.add_argument(
        "--tolerance_bp",
        type=int,
        default=5000,
        help="(bp), 5000",
    )
    args = p.parse_args()
    build_gt_all(DATA_DIR, args.output, tolerance_bp=args.tolerance_bp)
