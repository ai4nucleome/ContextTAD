#!/usr/bin/env python3
"""Measure cross-cell CTCF ChIP-seq support around left and right TAD boundaries."""

import os
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
EVAL_DATA_ROOT = Path(
    os.environ.get("EVAL_DATA_ROOT", PROJECT_ROOT / "0-data" / "2_eval_tads_data")
)
TOOLS_ROOT = Path(os.environ.get("EVAL_TOOLS_ROOT", EVAL_DATA_ROOT / "tools_output"))
DATA_DIR = Path(os.environ.get("EVAL_TOOLS_OTHERCELL_DIR", TOOLS_ROOT / "other_celltypes"))
OUTPUT_DIR = SCRIPT_DIR / "exp8_csv"
RESOLUTION = 5000
OFFSETS = list(range(-10, 11))

TAD_FILES = {
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
CTCF_FILES = {
    "K562": {
        "left": Path(
            os.environ.get(
                "EVAL_CTCF_LEFT_K562",
                EVAL_DATA_ROOT / "ctcf_chipseq" / "left" / "ENCFF901CBP_K562.bed_CTCF_5kb+.bed",
            )
        ),
        "right": Path(
            os.environ.get(
                "EVAL_CTCF_RIGHT_K562",
                EVAL_DATA_ROOT / "ctcf_chipseq" / "right" / "ENCFF901CBP_K562.bed_CTCF_5kb-.bed",
            )
        ),
    },
    "IMR90": {
        "left": Path(
            os.environ.get(
                "EVAL_CTCF_LEFT_IMR90",
                EVAL_DATA_ROOT / "ctcf_chipseq" / "left" / "ENCFF203SRF_IMR90.bed_CTCF_5kb+.bed",
            )
        ),
        "right": Path(
            os.environ.get(
                "EVAL_CTCF_RIGHT_IMR90",
                EVAL_DATA_ROOT / "ctcf_chipseq" / "right" / "ENCFF203SRF_IMR90.bed_CTCF_5kb-.bed",
            )
        ),
    },
}


def strip_chr(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace("chr", "", regex=False)


def load_ctcf_set(path: Path) -> dict[str, set[int]]:
    table = pd.read_csv(path, sep=r"\s+", header=None)
    table[0] = strip_chr(table[0])
    return {
        str(chrom): set(group[1].astype(int).tolist())
        for chrom, group in table.groupby(0, sort=False)
    }


def load_tads(cell: str, tool: str) -> pd.DataFrame:
    folder, bed_name = TAD_FILES[cell][tool]
    path = DATA_DIR / cell / folder / bed_name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=[0, 1, 2])
    table = pd.read_csv(path, sep="\t", header=None)[[0, 1, 2]].copy()
    table[0] = strip_chr(table[0])
    return table


def compute_hits(tads: pd.DataFrame, ctcf: dict[str, set[int]], column: int) -> np.ndarray:
    hits = np.zeros(len(OFFSETS), dtype=int)
    for chrom, position in tads[[0, column]].itertuples(index=False, name=None):
        chrom_sites = ctcf.get(str(chrom))
        if chrom_sites is None:
            continue
        position = int(position)
        for index, offset in enumerate(OFFSETS):
            if position + offset * RESOLUTION in chrom_sites:
                hits[index] += 1
    return hits


def main() -> None:
    rows = []
    for cell in ("K562", "IMR90"):
        left_ctcf = load_ctcf_set(CTCF_FILES[cell]["left"])
        right_ctcf = load_ctcf_set(CTCF_FILES[cell]["right"])
        for tool in TAD_FILES[cell]:
            tads = load_tads(cell, tool)
            left_hits = compute_hits(tads, left_ctcf, 1)
            right_hits = compute_hits(tads, right_ctcf, 2)
            for index, offset in enumerate(OFFSETS):
                rows.append(
                    {
                        "cell_line": cell,
                        "side": "left",
                        "tool": tool,
                        "offset_kb": offset * 5,
                        "count": int(left_hits[index]),
                    }
                )
                rows.append(
                    {
                        "cell_line": cell,
                        "side": "right",
                        "tool": tool,
                        "offset_kb": offset * 5,
                        "count": int(right_hits[index]),
                    }
                )

    OUTPUT_DIR.mkdir(exist_ok=True)
    result = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / "exp8_boundary_counts.csv"
    result.to_csv(out_csv, index=False)
    result.to_csv(OUTPUT_DIR / "ctcf_chipseq_boundary_counts.csv", index=False)
    print(out_csv)


if __name__ == "__main__":
    main()
