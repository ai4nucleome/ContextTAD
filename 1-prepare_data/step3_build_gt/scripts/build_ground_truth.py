#!/usr/bin/env python3
"""
Build test-split ground-truth BED from per-window `tads.npy`.

Windows are merged into chromosome-level intervals and deduplicated
with a configurable endpoint tolerance (default 5 kb).
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(os.environ.get("TAD_DATA_DIR", "/home/weicai/projectnvme/TADAnno_final/0-data/1_dp_train_infer_data"))
RESOLUTION = 5000
TEST_CHROMS = ["chr15", "chr16", "chr17"]
# TEST_CHROMS = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6",
#                "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14",
#                  "chr15", "chr16", "chr17",
#                  "chr18", "chr19", "chr20", "chr21", "chr22"]


def build_gt(data_dir, output_path, tolerance_bp=5000):
    with open(data_dir / "window_list.json") as f:
        wl = json.load(f)
    test_windows = wl["windows"]["test"]

    raw_tads = defaultdict(list)  # {chrom: [(start_bp, end_bp), ...]}

    for wn in test_windows:
        chrom = wn[:wn.rfind("_")]
        offset = int(wn[wn.rfind("_") + 1:])

        tads_path = data_dir / "labels" / chrom / f"{wn}_tads.npy"
        if not tads_path.exists():
            continue
        tads = np.load(tads_path)
        valid = tads[tads[:, 0] >= 0]

        for left, right in valid:
            start_bp = (offset + int(left)) * RESOLUTION
            end_bp = (offset + int(right) + 1) * RESOLUTION
            raw_tads[chrom].append((start_bp, end_bp))

    # Deduplicate near-identical intervals by endpoint tolerance.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(output_path, "w") as f:
        for chrom in sorted(raw_tads):
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

            print(f"  {chrom}: {len(raw_tads[chrom])} raw  {len(deduped)} deduped")

    print(f"Total: {total} TADs  {output_path}")
    return total


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "ground_truth_TAD_dp_test.bed"),
    )
    args = p.parse_args()
    build_gt(DATA_DIR, args.output)
