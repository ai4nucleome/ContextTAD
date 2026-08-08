#!/usr/bin/env python3
"""Compute 250M TAD-size summary statistics for all benchmark callers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
EVAL_DATA_ROOT = Path(
    os.environ.get("EVAL_DATA_ROOT", PROJECT_ROOT / "0-data/2_eval_tads_data")
)
TOOLS_ROOT = Path(os.environ.get("EVAL_TOOLS_ROOT", EVAL_DATA_ROOT / "tools_output"))
COVERAGE = os.environ.get("COVERAGE", "250M")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", SCRIPT_DIR / "exp9_csv"))
MAX_SIZE_BP = int(os.environ.get("MAX_SIZE_BP", "3000000"))

TOOL_PATHS = {
    "Armatus": ("Armatus", "4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_Armatus.bed"),
    "Arrowhead": ("Arrowhead", "4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_Arrowhead.bed"),
    "CaTCH": ("CaTCH", "4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_CaTCH.bed"),
    "Domaincall": ("Domaincall", "4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_DI.bed"),
    "EAST": ("EAST", "4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_EAST2.bed"),
    "Grinch": ("Grinch", "4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_grinch.bed"),
    "HiCSeg": ("HiCSeg", "4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_HiCSeg.bed"),
    "IC-Finder": ("IC-Finder", "4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_icfinder.bed"),
    "OnTAD": ("OnTAD", "4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_OnTAD.bed"),
    "GMAP": ("GMAP", "4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_rGMAP.bed"),
    "TopDom": ("TopDom", "4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_TopDom.bed"),
    "RefHiC": ("RefHiC", "4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_refhic.bed"),
    "deDoc": ("deDoc", "4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_deDoc.deDocM.bed"),
    "HiTAD": ("HiTAD", "4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_hitad.bed"),
    "RobusTAD": ("RobusTAD", "4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_robustad.good.bed"),
    "ContextTAD": ("ContextTAD", "4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_contexttad.bed"),
}


def load_sizes(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing BED file: {path}")
    if path.stat().st_size == 0:
        return np.empty(0, dtype=float)

    bed = pd.read_csv(path, sep="\t", header=None, usecols=[1, 2])
    sizes = pd.to_numeric(bed[2], errors="coerce") - pd.to_numeric(
        bed[1], errors="coerce"
    )
    sizes = sizes[(sizes > 0) & (sizes < MAX_SIZE_BP)].dropna()
    # Floor division reproduces the original benchmark's bp-to-kb conversion.
    return sizes.to_numpy(dtype=float) // 1000


def summarize(tool: str, sizes: np.ndarray) -> dict[str, float | int | str]:
    if sizes.size == 0:
        return {
            "tool": tool,
            "n_tads": 0,
            "mean_kb": 0.0,
            "median_kb": 0.0,
            "std_kb": 0.0,
            "q1_kb": 0.0,
            "q3_kb": 0.0,
            "min_kb": 0.0,
            "max_kb": 0.0,
        }
    return {
        "tool": tool,
        "n_tads": int(sizes.size),
        "mean_kb": float(np.mean(sizes)),
        "median_kb": float(np.median(sizes)),
        "std_kb": float(np.std(sizes)),
        "q1_kb": float(np.quantile(sizes, 0.25)),
        "q3_kb": float(np.quantile(sizes, 0.75)),
        "min_kb": float(np.min(sizes)),
        "max_kb": float(np.max(sizes)),
    }


def main() -> None:
    coverage_root = TOOLS_ROOT / COVERAGE
    rows = []
    for tool, (folder, filename) in TOOL_PATHS.items():
        rows.append(summarize(tool, load_sizes(coverage_root / folder / filename)))

    result = pd.DataFrame(rows).sort_values(
        ["median_kb", "mean_kb", "tool"], ignore_index=True
    )
    result.insert(0, "median_rank", np.arange(1, len(result) + 1))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "exp9_tad_size_results.csv"
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"[output] {output}")


if __name__ == "__main__":
    main()
