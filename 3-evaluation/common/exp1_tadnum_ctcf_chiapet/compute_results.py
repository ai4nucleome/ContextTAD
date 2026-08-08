#!/usr/bin/env python3
"""Aggregate GM12878 250M TAD counts and CTCF ChIA-PET support."""

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
TOOLS_ROOT = Path(os.environ.get("EVAL_TOOLS_ROOT", EVAL_DATA_ROOT / "tools_output"))
DATA_DIR = Path(os.environ.get("EVAL_TOOLS_250M_DIR", TOOLS_ROOT / "250M"))
CTCF_DATA = Path(
    os.environ.get(
        "EVAL_CTCF_CHIAPET_FILE",
        EVAL_DATA_ROOT / "ctcf_chiapet" / "gm12878.tang.ctcf-chiapet.hg38.bedpe",
    )
)
OUTPUT_DIR = SCRIPT_DIR / "exp1_csv"
RESOLUTION = 5000

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
    "ContextTAD": ("ContextTAD", "4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_contexttad.bed"),
    "RobusTAD": ("RobusTAD", "4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_robustad.good.bed"),
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

    rows = []
    for tool, (folder, bed_base) in TOOL_PATHS.items():
        l0_supported, l0_unsupported = count_supported_tads(
            CTCF_DATA, DATA_DIR / folder / f"{bed_base}.L0"
        )
        l1_supported, l1_unsupported = count_supported_tads(
            CTCF_DATA, DATA_DIR / folder / f"{bed_base}.L1+"
        )
        rows.append(
            {
                "tool": tool,
                "CTCF supported level 0 TAD": l0_supported,
                "CTCF supported level 1+ TAD": l1_supported,
                "unsupported level 0 TAD": l0_unsupported,
                "unsupported level 1+ TAD": l1_unsupported,
            }
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_csv = OUTPUT_DIR / "exp1_results.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(out_csv)


if __name__ == "__main__":
    main()
