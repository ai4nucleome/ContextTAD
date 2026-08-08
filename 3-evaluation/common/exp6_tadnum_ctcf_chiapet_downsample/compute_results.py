#!/usr/bin/env python3
"""Aggregate multi-coverage TAD counts and CTCF ChIA-PET support."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
EVAL_DATA_ROOT = Path(
    os.environ.get("EVAL_DATA_ROOT", PROJECT_ROOT / "0-data" / "2_eval_tads_data")
)
DATA_ROOT = Path(os.environ.get("EVAL_TOOLS_ROOT", EVAL_DATA_ROOT / "tools_output"))
CTCF_DATA = Path(
    os.environ.get(
        "EVAL_CTCF_CHIAPET_FILE",
        EVAL_DATA_ROOT / "ctcf_chiapet" / "gm12878.tang.ctcf-chiapet.hg38.bedpe",
    )
)
OUTPUT_DIR = SCRIPT_DIR / "exp6_csv"
RESOLUTION = 5000

TOOL_PATHS = {
    "Armatus": ("Armatus", "4DNFIXP4QG5B_Rao2014_GM12878_{cov}_5K_Armatus.bed"),
    "Arrowhead": ("Arrowhead", "4DNFIXP4QG5B_Rao2014_GM12878_{cov}_5K_Arrowhead.bed"),
    "CaTCH": ("CaTCH", "4DNFIXP4QG5B_Rao2014_GM12878_{cov}_5K_CaTCH.bed"),
    "Domaincall": ("Domaincall", "4DNFIXP4QG5B_Rao2014_GM12878_{cov}_5K_DI.bed"),
    "EAST": ("EAST", "4DNFIXP4QG5B_Rao2014_GM12878_{cov}_5K_EAST2.bed"),
    "Grinch": ("Grinch", "4DNFIXP4QG5B_Rao2014_GM12878_{cov}_5K_grinch.bed"),
    "HiCSeg": ("HiCSeg", "4DNFIXP4QG5B_Rao2014_GM12878_{cov}_5K_HiCSeg.bed"),
    "IC-Finder": ("IC-Finder", "4DNFIXP4QG5B_Rao2014_GM12878_{cov}_5K_icfinder.bed"),
    "OnTAD": ("OnTAD", "4DNFIXP4QG5B_Rao2014_GM12878_{cov}_5K_OnTAD.bed"),
    "GMAP": ("GMAP", "4DNFIXP4QG5B_Rao2014_GM12878_{cov}_5K_rGMAP.bed"),
    "TopDom": ("TopDom", "4DNFIXP4QG5B_Rao2014_GM12878_{cov}_5K_TopDom.bed"),
    "RefHiC": ("RefHiC", "4DNFIXP4QG5B_Rao2014_GM12878_{cov}_5K_refhic.bed"),
    "deDoc": ("deDoc", "4DNFIXP4QG5B_Rao2014_GM12878_{cov}_5K_deDoc.deDocM.bed"),
    "HiTAD": ("HiTAD", "4DNFIXP4QG5B_Rao2014_GM12878_{cov}_5K_hitad.bed"),
    "ContextTAD": ("ContextTAD", "4DNFIXP4QG5B_Rao2014_GM12878_{cov}_5K_contexttad.bed"),
    "RobusTAD": ("RobusTAD", "4DNFIXP4QG5B_Rao2014_GM12878_{cov}_5K_robustad.good.bed"),
}
COVERAGES = {
    "4000M": "4000M",
    "2000M": "2000M",
    "1000M": "1000M",
    "500M": "500M",
    "250M": "250M",
    "125M": "125M",
    "62_5M": "62.5M",
}


def count_supported_tads(target_file: Path, pred_file: Path) -> tuple[int, int]:
    if not pred_file.exists() or pred_file.stat().st_size == 0:
        return 0, 0

    target = pd.read_csv(target_file, header=None, sep="\t", comment="#")
    pred = pd.read_csv(pred_file, header=None, sep="\t", comment="#")
    target[[1, 2, 4, 5]] = target[[1, 2, 4, 5]] // RESOLUTION
    target[[2, 5]] = target[[1, 4]] + 1
    pred[4] = pred[2]
    pred[2] = pred[1] + RESOLUTION
    pred[5] = pred[4] + RESOLUTION
    pred[[1, 2, 4, 5]] = pred[[1, 2, 4, 5]] // RESOLUTION
    target = target.drop_duplicates()
    pred = pred.drop_duplicates()
    pred["hits"] = 0

    for chrom in set(pred[0]):
        target_chrom = target[target[0] == chrom]
        pred_chrom = pred[pred[0] == chrom]
        if target_chrom.empty or pred_chrom.empty:
            continue
        tree = KDTree(target_chrom[[1, 4]].to_numpy(), leaf_size=30, metric="chebyshev")
        neighbors = tree.query_radius(pred_chrom[[1, 4]].to_numpy(), r=1, return_distance=False)
        pred.loc[pred[0] == chrom, "hits"] = np.fromiter(
            (len(indices) > 0 for indices in neighbors), dtype=int
        )

    supported = int(pred["hits"].sum())
    return supported, len(pred) - supported


def main() -> None:
    if not CTCF_DATA.exists():
        raise FileNotFoundError(f"missing CTCF ChIA-PET file: {CTCF_DATA}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    frames = []
    for coverage_dir, coverage_name in COVERAGES.items():
        rows = []
        for tool, (folder, bed_template) in TOOL_PATHS.items():
            bed_base = bed_template.format(cov=coverage_name)
            tool_dir = DATA_ROOT / coverage_dir / folder
            l0_supported, l0_unsupported = count_supported_tads(
                CTCF_DATA, tool_dir / f"{bed_base}.L0"
            )
            l1_supported, l1_unsupported = count_supported_tads(
                CTCF_DATA, tool_dir / f"{bed_base}.L1+"
            )
            rows.append(
                {
                    "cov": coverage_name,
                    "tool": tool,
                    "CTCF supported level 0 TAD": l0_supported,
                    "CTCF supported level 1+ TAD": l1_supported,
                    "unsupported level 0 TAD": l0_unsupported,
                    "unsupported level 1+ TAD": l1_unsupported,
                }
            )
        frame = pd.DataFrame(rows)
        frames.append(frame)
        frame.to_csv(OUTPUT_DIR / f"exp6_results_{coverage_name}.csv", index=False)
        frame.to_csv(OUTPUT_DIR / f"exp14_results_{coverage_name}.csv", index=False)

    summary = pd.concat(frames, ignore_index=True)
    out_csv = OUTPUT_DIR / "exp6_results_all.csv"
    summary.to_csv(out_csv, index=False)
    summary.to_csv(OUTPUT_DIR / "exp14_results_all.csv", index=False)
    print(out_csv)


if __name__ == "__main__":
    main()
