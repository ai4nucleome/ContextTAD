#!/usr/bin/env python3
"""Create fixed-width chromosome bins for the histone permutation test."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


# Retained from the published TAD benchmarking implementation for exact
# methodological compatibility.
CHROM_LENGTHS = {
    15: 102_531_392,
    16: 90_354_753,
    17: 81_195_210,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outFile", required=True)
    parser.add_argument("-c", "--chrom", required=True, type=int)
    parser.add_argument("-b", "--binsize", required=True, type=int)
    args = parser.parse_args()

    if args.binsize <= 0:
        raise ValueError("binsize must be positive")
    if args.chrom not in CHROM_LENGTHS:
        raise ValueError(f"unsupported chromosome: {args.chrom}")

    chrom_len = CHROM_LENGTHS[args.chrom]
    n_bins = math.ceil(chrom_len / args.binsize)
    output = Path(args.outFile)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for index in range(n_bins):
            start = 1 + index * args.binsize
            end = min((index + 1) * args.binsize, chrom_len)
            handle.write(f"chr{args.chrom}\t{start}\t{end}\t+\n")


if __name__ == "__main__":
    main()
