#!/usr/bin/env python3
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
rows = []
for d in sorted(OUT_DIR.iterdir()):
    if not d.is_dir():
        continue
    sm = d / "eval_results" / "summary_metrics.csv"
    if not sm.exists():
        continue
    with sm.open(newline="") as f:
        r = next(csv.DictReader(f))
    rows.append(
        {
            "run": d.name,
            "exp1_total_supported": int(r["exp1_total_supported"]),
            "exp7_left_fraction": float(r["exp7_left_fraction"]),
            "exp8_right_fraction": float(r["exp8_right_fraction"]),
            "exp14_total_supported_all_cov": int(r["exp14_total_supported_all_cov"]),
        }
    )

rows.sort(key=lambda x: x["run"])
out_csv = ROOT / "outputs" / "model_ablation_summary.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "run",
            "exp1_total_supported",
            "exp7_left_fraction",
            "exp8_right_fraction",
            "exp14_total_supported_all_cov",
        ],
    )
    w.writeheader()
    w.writerows(rows)
print(out_csv)
