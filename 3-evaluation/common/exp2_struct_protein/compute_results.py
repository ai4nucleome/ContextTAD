#!/usr/bin/env python3
"""Aggregate CTCF, RAD21, and SMC3 fold enrichment."""

import os
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = Path(
    os.environ.get(
        "EXP6_DATA_FILE", SCRIPT_DIR / "struct_protein" / "StructProteins_allTools.tsv"
    )
)
OUTPUT_DIR = Path(os.environ.get("EXP6_OUTPUT_DIR", SCRIPT_DIR / "exp2_csv"))
PROTEINS = ["CTCF", "RAD21", "SMC3"]


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"missing structural-protein result table: {DATA_FILE}")

    source = pd.read_csv(DATA_FILE, sep="\t")
    required = {"TadsFile", "protein", "fc_over_bg"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"missing columns in {DATA_FILE}: {sorted(missing)}")

    result = (
        source.pivot_table(index="TadsFile", columns="protein", values="fc_over_bg", aggfunc="first")
        .reindex(columns=PROTEINS)
        .dropna()
        .reset_index()
    )
    result["mean"] = result[PROTEINS].sum(axis=1)
    result = result.sort_values("mean").rename(columns={"TadsFile": "tool"})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "exp2_results.csv"
    result.to_csv(out_csv, index=False)
    result.to_csv(OUTPUT_DIR / "exp6_results.csv", index=False)
    print(out_csv)


if __name__ == "__main__":
    main()
