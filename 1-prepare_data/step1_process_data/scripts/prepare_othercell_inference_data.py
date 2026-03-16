#!/usr/bin/env python3
"""
Prepare other-cell inference data (windowed .npy + window_list.json) from an mcool file.

Output layout:
  <out_data_dir>/
    window_list.json
    <coverage_tag>/
      chr1/chr1_0.npy
      chr1/chr1_200.npy
      ...

Each npy stores stacked matrices: [2, 400, 400]
  channel 0 = balanced obs
  channel 1 = oe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import cooler
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import tqdm


def upper_coo_to_symm(row: np.ndarray, col: np.ndarray, data: np.ndarray, n: int):
    mat = coo_matrix((data, (row, col)), shape=(n, n))
    sym = mat + mat.T
    diag = sym.diagonal(0) / 2
    sym = sym.tocsr()
    sym.setdiag(diag)
    return sym


def process_chrom(clr: cooler.Cooler, chrom: str):
    extent = clr.extent(chrom)
    n = extent[1] - extent[0]
    pix = clr.matrix(balance=True, sparse=True, as_pixels=True).fetch(chrom)
    if len(pix) == 0:
        return None, None, n

    pix = pix.copy()
    pix["bin1_id"] -= extent[0]
    pix["bin2_id"] -= extent[0]
    pix["distance"] = pix["bin2_id"] - pix["bin1_id"]
    means = pix.groupby("distance")["balanced"].transform("mean")
    bal = pix["balanced"].fillna(0)
    oe = (pix["balanced"] / means).replace([np.inf, -np.inf], 0).fillna(0)

    row = pix["bin1_id"].to_numpy(dtype=np.int32)
    col = pix["bin2_id"].to_numpy(dtype=np.int32)
    obs_mat = upper_coo_to_symm(row, col, bal.to_numpy(dtype=np.float32), n)
    oe_mat = upper_coo_to_symm(row, col, oe.to_numpy(dtype=np.float32), n)
    return obs_mat, oe_mat, n


def get_positions(n: int, window_size: int, step_size: int) -> List[int]:
    starts: List[int] = []
    seen = set()
    for start in range(0, n, step_size):
        if start + window_size > n:
            start = max(0, n - window_size)
        if start not in seen:
            starts.append(start)
            seen.add(start)
    if not starts:
        starts = [0]
    return starts


def extract_square(mat, n: int, start: int, window_size: int) -> np.ndarray:
    end = min(n, start + window_size)
    arr = mat[start:end, start:end].toarray()
    if arr.shape[0] < window_size:
        pad = window_size - arr.shape[0]
        arr = np.pad(arr, ((0, pad), (0, pad)), mode="constant")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32)


def resolve_mcool_uri(mcool_path: str, resolution: int) -> str:
    if "::/resolutions/" in mcool_path:
        return mcool_path
    return f"{mcool_path}::/resolutions/{resolution}"


def parse_args():
    p = argparse.ArgumentParser(description="Prepare other-cell inference windows from mcool")
    p.add_argument("--mcool", required=True, help="Path to mcool (with or without ::/resolutions/<res>)")
    p.add_argument("--out_data_dir", required=True, help="Output root data directory")
    p.add_argument("--coverage_tag", required=True, help="Coverage directory name used by predictor")
    p.add_argument("--resolution", type=int, default=5000)
    p.add_argument("--window_size", type=int, default=400)
    p.add_argument("--step_size", type=int, default=200)
    p.add_argument("--chroms", nargs="+", default=[f"chr{i}" for i in range(1, 23)])
    return p.parse_args()


def main():
    args = parse_args()
    if args.step_size > args.window_size:
        raise ValueError("step_size must be <= window_size")

    mcool_uri = resolve_mcool_uri(args.mcool, args.resolution)
    out_data_dir = Path(args.out_data_dir)
    cov_root = out_data_dir / args.coverage_tag
    cov_root.mkdir(parents=True, exist_ok=True)

    clr = cooler.Cooler(mcool_uri)
    available = set(clr.chromnames)
    test_windows: List[str] = []
    kept_chroms: List[str] = []

    for chrom in args.chroms:
        if chrom not in available:
            continue
        obs_mat, oe_mat, n = process_chrom(clr, chrom)
        if obs_mat is None or oe_mat is None:
            continue

        kept_chroms.append(chrom)
        chrom_dir = cov_root / chrom
        chrom_dir.mkdir(parents=True, exist_ok=True)

        positions = get_positions(n, args.window_size, args.step_size)
        for start in tqdm(positions, desc=f"prepare {chrom}"):
            win_name = f"{chrom}_{start}"
            obs_small = extract_square(obs_mat, n, start, args.window_size)
            oe_small = extract_square(oe_mat, n, start, args.window_size)
            mats = np.stack([obs_small, oe_small], axis=0)
            np.save(chrom_dir / f"{win_name}.npy", mats)
            test_windows.append(win_name)

    test_windows = sorted(test_windows, key=lambda x: (x.split("_")[0], int(x.split("_")[1])))
    wl = {
        "windows": {"train": [], "val": [], "test": test_windows},
        "total_windows": len(test_windows),
        "chroms": kept_chroms,
        "coverage_tag": args.coverage_tag,
        "mcool": str(args.mcool),
        "resolution": args.resolution,
        "window_size": args.window_size,
        "step_size": args.step_size,
    }
    (out_data_dir / "window_list.json").write_text(json.dumps(wl, indent=2))

    print(f"prepared_windows={len(test_windows)}")
    print(f"output_root={out_data_dir}")
    print(f"coverage_tag={args.coverage_tag}")


if __name__ == "__main__":
    main()
