#!/usr/bin/env python3
"""Combine chromosome-level histone results with TAD-count weighting."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    files = sorted(
        args.output_dir.glob("Results_df_shuf*_share*_fdrthresh*_chr*.txt")
    )
    if not files:
        raise FileNotFoundError(f"no chromosome-level results in {args.output_dir}")

    total_tads = 0
    significant = 0.0
    binsizes = []
    for path in files:
        with path.open() as handle:
            row = next(csv.DictReader(handle, delimiter="\t"))
        count = int(row["nTads"])
        total_tads += count
        significant += count * float(row["shareTadsSignif_lr_BH"])
        binsizes.append(int(float(row["binsize"])))

    fraction = significant / total_tads if total_tads else 0.0
    prefix = files[0].name.split("_chr", maxsplit=1)[0]
    output = args.output_dir / f"{prefix}_mean.txt"
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["nTads", "binsize", "shareTadsSignif_lr_BH"])
        # Preserve the benchmark output schema; this field records the final
        # chromosome's bin size and is not used in the cross-tool summary.
        writer.writerow([total_tads, binsizes[-1], fraction])
    print(output)


if __name__ == "__main__":
    main()
