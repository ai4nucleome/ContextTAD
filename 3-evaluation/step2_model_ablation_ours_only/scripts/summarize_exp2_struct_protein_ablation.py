#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROTEINS = ["CTCF", "RAD21", "SMC3"]
DISPLAY_NAME = {
    "base": "Base",
    "no_tofe": "No TOFE",
    "no_text": "No Text",
    "no_pairloss": "No PairLoss",
    "obs_input": "Obs Input",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot(index="TadsFile", columns="protein", values="fc_over_bg").reset_index()
    missing = [protein for protein in PROTEINS if protein not in pivot.columns]
    if missing:
        raise ValueError(f"missing proteins: {missing}")

    pivot = pivot.rename(columns={"TadsFile": "method"})
    pivot["display_name"] = pivot["method"].map(DISPLAY_NAME).fillna(pivot["method"])
    pivot["mean"] = pivot[PROTEINS].sum(axis=1)
    pivot = pivot.sort_values(["mean", "display_name"], ascending=[False, True]).reset_index(drop=True)
    pivot["rank_by_mean"] = pivot["mean"].rank(method="min", ascending=False).astype(int)
    cols = ["rank_by_mean", "method", "display_name", *PROTEINS, "mean"]
    return pivot[cols]


def write_markdown(df: pd.DataFrame, output_md: Path) -> None:
    lines = [
        "# exp2 struct protein results",
        "",
        "| Rank | Method | CTCF | RAD21 | SMC3 | Mean |",
        "| ---: | :----- | ---: | ----: | ---: | ---: |",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.rank_by_mean} | {row.display_name} | "
            f"{row.CTCF:.6f} | {row.RAD21:.6f} | {row.SMC3:.6f} | {row.mean:.6f} |"
        )
    output_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_tsv, sep="\t")
    summary = build_summary(df)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    write_markdown(summary, args.output_md)
    print(args.output_csv)
    print(args.output_md)


if __name__ == "__main__":
    main()
