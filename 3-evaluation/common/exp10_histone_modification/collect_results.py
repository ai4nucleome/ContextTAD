#!/usr/bin/env python3
"""Collect per-tool histone-modification permutation-test results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tools", nargs="+", required=True)
    args = parser.parse_args()

    rows = []
    selected_tools = set(args.tools)
    for mean_file in sorted(
        args.results_dir.glob("*_HistMod/Results_df_shuf*_share*_fdrthresh*_mean.txt")
    ):
        with mean_file.open() as handle:
            record = next(csv.DictReader(handle, delimiter="\t"))
        tool = mean_file.parent.name.removesuffix("_HistMod")
        if tool not in selected_tools:
            continue
        n_tads = int(record["nTads"])
        fraction = float(record["shareTadsSignif_lr_BH"])
        rows.append(
            {
                "tool": tool,
                "n_tads": n_tads,
                "significant_tad_count": round(n_tads * fraction),
                "significant_fraction": fraction,
            }
        )

    if not rows:
        raise FileNotFoundError(f"no mean result files under {args.results_dir}")
    missing = selected_tools.difference(row["tool"] for row in rows)
    if missing:
        raise FileNotFoundError(f"missing mean results for: {sorted(missing)}")

    result = pd.DataFrame(rows).sort_values(
        ["significant_fraction", "significant_tad_count", "tool"],
        ascending=[False, False, True],
        ignore_index=True,
    )
    result.insert(0, "fraction_rank", range(1, len(result) + 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"[output] {args.output}")


if __name__ == "__main__":
    main()
