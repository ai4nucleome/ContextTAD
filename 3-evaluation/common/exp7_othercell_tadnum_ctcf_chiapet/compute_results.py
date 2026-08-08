#!/usr/bin/env python3
"""Aggregate cross-cell TAD counts and CTCF ChIA-PET support."""

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
DATA_DIR = Path(os.environ.get("EVAL_TOOLS_OTHERCELL_DIR", TOOLS_ROOT / "other_celltypes"))
OUTPUT_DIR = SCRIPT_DIR / "exp7_csv"
RESOLUTION = 5000

CTCF_DATA = {
    "K562": Path(
        os.environ.get(
            "EVAL_CTCF_CHIAPET_K562",
            EVAL_DATA_ROOT / "ctcf_chiapet" / "k562.encode.ctcf-chiapet.5k.hg38.bedpe",
        )
    ),
    "IMR90": Path(
        os.environ.get(
            "EVAL_CTCF_CHIAPET_IMR90",
            EVAL_DATA_ROOT / "ctcf_chiapet" / "imr90_ctcf_chiapet_hg38_ENCFF682YFU.bedpe",
        )
    ),
}
TOOL_PATHS = {
    "K562": {
        "Arrowhead": ("Arrowhead", "K562_Arrowhead.bed"),
        "GMAP": ("GMAP", "K562_rGMAP.bed"),
        "HiTAD": ("HiTAD", "K562_hitad.bed"),
        "OnTAD": ("OnTAD", "K562_OnTAD.bed"),
        "RefHiC": ("RefHic", "K562_RefHiC.bed"),
        "ContextTAD": ("ContextTAD", "K562_contexttad.bed"),
        "RobusTAD": ("RobusTAD", "K562_robustad_all.good.bed"),
    },
    "IMR90": {
        "Arrowhead": ("Arrowhead", "IMR90_Arrowhead.bed"),
        "GMAP": ("GMAP", "IMR90_rGMAP.bed"),
        "HiTAD": ("HiTAD", "IMR90_hitad.bed"),
        "OnTAD": ("OnTAD", "IMR90_OnTAD.bed"),
        "RefHiC": ("RefHic", "IMR90_RefHiC.bed"),
        "ContextTAD": ("ContextTAD", "IMR90_contexttad.bed"),
        "RobusTAD": ("RobusTAD", "IMR90_robustad_all.good.bed"),
    },
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
    for cell, path in CTCF_DATA.items():
        if not path.exists():
            raise FileNotFoundError(f"missing CTCF ChIA-PET file for {cell}: {path}")

    rows = []
    for cell in ("K562", "IMR90"):
        for tool, (folder, bed_base) in TOOL_PATHS[cell].items():
            tool_dir = DATA_DIR / cell / folder
            l0_supported, l0_unsupported = count_supported_tads(
                CTCF_DATA[cell], tool_dir / f"{bed_base}.L0"
            )
            l1_supported, l1_unsupported = count_supported_tads(
                CTCF_DATA[cell], tool_dir / f"{bed_base}.L1+"
            )
            rows.append(
                {
                    "cell_line": cell,
                    "tool": tool,
                    "CTCF supported level 0 TAD": l0_supported,
                    "CTCF supported level 1+ TAD": l1_supported,
                    "unsupported level 0 TAD": l0_unsupported,
                    "unsupported level 1+ TAD": l1_unsupported,
                }
            )

    OUTPUT_DIR.mkdir(exist_ok=True)
    result = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / "exp7_results.csv"
    result.to_csv(out_csv, index=False)
    result.to_csv(OUTPUT_DIR / "exp16_results.csv", index=False)
    print(out_csv)


if __name__ == "__main__":
    main()
