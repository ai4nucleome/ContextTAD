#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import zlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from scipy.signal import find_peaks

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from model import ContextTADBackboneModel  # type: ignore
from train import normalize_matrix  # type: ignore

PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_DIR = Path(
    os.environ.get("TAD_DATA_DIR", str(PROJECT_ROOT / "0-data" / "1_dp_train_infer_data"))
)
RESOLUTION = 5000
MIN_LEN_BINS = 5
MAX_LEN_BINS = 350
NMS_METRIC = "iou"
NMS_THRESH = 0.5
ALLOW_NESTED = True
W_LEFT = 1.0
W_RIGHT = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_seed(text: str, base: int = 66) -> int:
    return (base + zlib.adler32(text.encode("utf-8"))) % (2**32 - 1)


def diagonal_preserving_decoy(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle each genomic-distance diagonal while preserving symmetry."""
    x = np.asarray(arr, dtype=np.float32)
    n = x.shape[0]
    out = np.zeros_like(x, dtype=np.float32)
    diag = np.diag(x).copy()
    rng.shuffle(diag)
    np.fill_diagonal(out, diag)
    for d in range(1, n):
        vals = np.diag(x, k=d).copy()
        rng.shuffle(vals)
        i = np.arange(n - d)
        out[i, i + d] = vals
        out[i + d, i] = vals
    return out


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    backbone = args.get("backbone", "unet_decoder")
    hidden = int(args.get("hidden", 128))
    diag_radius = int(args.get("diag_radius", 50))
    model = ContextTADBackboneModel(hidden=hidden, diag_radius=diag_radius, backbone=backbone or "unet_decoder")
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()
    return model, args


def fdr_cutoff(target: np.ndarray, decoy: np.ndarray, alpha: float) -> float:
    target = np.asarray(target, dtype=np.float32)
    decoy = np.asarray(decoy, dtype=np.float32)
    if len(target) == 0:
        return float("inf")
    if len(decoy) == 0:
        return float(np.min(target))
    vals = np.concatenate([target, decoy])
    labels = np.concatenate([np.ones(len(target), dtype=bool), np.zeros(len(decoy), dtype=bool)])
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    labels = labels[order]
    nt = 0
    nd = 0
    cutoff = float("inf")
    eps = np.finfo(float).eps
    for val, is_target in zip(vals, labels):
        if is_target:
            nt += 1
        else:
            nd += 1
        if nt > 0 and (nd / (nt + eps)) <= alpha:
            cutoff = float(val)
    return cutoff


def peak_metrics(scores: np.ndarray, peak_radius: int, metric: str) -> Tuple[np.ndarray, np.ndarray]:
    peaks, props = find_peaks(scores, distance=peak_radius * 2 + 1, prominence=(None, None))
    peaks = peaks.astype(np.int32)
    if len(peaks) == 0:
        return peaks, np.zeros(0, dtype=np.float32)
    if metric == "height":
        values = scores[peaks].astype(np.float32)
    elif metric == "prominence":
        values = props.get("prominences", np.zeros(len(peaks), dtype=np.float32)).astype(np.float32)
    else:
        raise ValueError(f"Unknown fdr metric: {metric}")
    return peaks, values


def select_fdr_peaks(target_scores: np.ndarray, decoy_scores: np.ndarray, peak_radius: int, alpha: float, metric: str) -> Tuple[np.ndarray, float, int, int]:
    target_peaks, target_values = peak_metrics(target_scores, peak_radius, metric)
    decoy_peaks, decoy_values = peak_metrics(decoy_scores, peak_radius, metric)
    cutoff = fdr_cutoff(target_values, decoy_values, alpha)
    if len(target_peaks) == 0 or not np.isfinite(cutoff):
        return np.zeros(0, dtype=np.int32), cutoff, int(len(target_peaks)), int(len(decoy_peaks))
    selected = target_peaks[target_values >= cutoff]
    return selected.astype(np.int32), cutoff, int(len(target_peaks)), int(len(decoy_peaks))


def pair_boundaries(
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    left_candidates: np.ndarray,
    right_candidates: np.ndarray,
    min_len_bins: int,
    max_len_bins: int,
    nms_metric: str,
    nms_thresh: float,
    allow_nested: bool,
) -> List[Tuple[int, int, float]]:
    left_candidates = np.asarray(left_candidates, dtype=np.int32)
    right_candidates = np.asarray(right_candidates, dtype=np.int32)
    left_candidates.sort()
    right_candidates.sort()
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        return []

    left_prob_all = np.clip(left_scores[left_candidates], 1e-6, 1.0 - 1e-6).astype(np.float32)
    right_prob_all = np.clip(right_scores[right_candidates], 1e-6, 1.0 - 1e-6).astype(np.float32)

    l_chunks = []
    r_chunks = []
    score_chunks = []
    for l, l_prob in zip(left_candidates, left_prob_all):
        lo = int(l) + min_len_bins - 1
        hi = int(l) + max_len_bins - 1
        a = int(np.searchsorted(right_candidates, lo, side="left"))
        b = int(np.searchsorted(right_candidates, hi, side="right"))
        if b <= a:
            continue
        rs = right_candidates[a:b]
        scores = float(l_prob) * right_prob_all[a:b]
        l_chunks.append(np.full(len(rs), int(l), dtype=np.int32))
        r_chunks.append(rs.astype(np.int32, copy=False))
        score_chunks.append(scores.astype(np.float32, copy=False))

    if not l_chunks:
        return []

    cand_l = np.concatenate(l_chunks)
    cand_r = np.concatenate(r_chunks)
    cand_scores = np.concatenate(score_chunks)
    order = np.argsort(cand_scores)[::-1]
    max_select = len(cand_scores)

    sel_l = np.empty(max_select, dtype=np.int32)
    sel_r = np.empty(max_select, dtype=np.int32)
    sel_score = np.empty(max_select, dtype=np.float32)
    nsel = 0

    for idx in order:
        l = int(cand_l[idx])
        r = int(cand_r[idx])
        if nsel:
            sl = sel_l[:nsel]
            sr = sel_r[:nsel]
            inter = np.minimum(r, sr) - np.maximum(l, sl) + 1
            overlap_mask = inter > 0
            if overlap_mask.any():
                if allow_nested:
                    contained = ((l >= sl) & (r <= sr)) | ((sl >= l) & (sr <= r))
                    overlap_mask = overlap_mask & (~contained)
                if overlap_mask.any():
                    len1 = r - l + 1
                    len2 = sr[overlap_mask] - sl[overlap_mask] + 1
                    inter2 = inter[overlap_mask].astype(np.float32)
                    if nms_metric == "iou":
                        suppress = inter2 / np.maximum(len1 + len2 - inter2, 1.0)
                    else:
                        suppress = inter2 / np.maximum(np.minimum(len1, len2), 1.0)
                    if np.any(suppress > nms_thresh):
                        continue
        sel_l[nsel] = l
        sel_r[nsel] = r
        sel_score[nsel] = float(cand_scores[idx])
        nsel += 1
        if nsel >= max_select:
            break

    return [(int(sel_l[i]), int(sel_r[i]), float(sel_score[i])) for i in range(nsel)]


def aggregate_target_decoy_tracks(model, chrom: str, data_dir: Path, cov: str, use_oe: bool, device: torch.device, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cov_dir = data_dir / cov / chrom
    if not cov_dir.exists():
        z = np.zeros(0, dtype=np.float32)
        return z, z, z, z
    with (data_dir / "window_list.json").open() as f:
        wl = json.load(f)
    test_windows = sorted([w for w in wl["windows"]["test"] if w.startswith(chrom + "_")])
    if not test_windows:
        z = np.zeros(0, dtype=np.float32)
        return z, z, z, z
    chrom_size = max(int(w.split("_")[-1]) for w in test_windows) + 400
    lt = np.zeros(chrom_size, dtype=np.float32)
    rt = np.zeros(chrom_size, dtype=np.float32)
    ld = np.zeros(chrom_size, dtype=np.float32)
    rd = np.zeros(chrom_size, dtype=np.float32)
    count = np.zeros(chrom_size, dtype=np.float32)
    for wn in test_windows:
        offset = int(wn.split("_")[-1])
        npy_path = cov_dir / f"{wn}.npy"
        if not npy_path.exists():
            continue
        mat = np.load(npy_path)
        arr = mat[1] if use_oe else mat[0]
        rng = np.random.default_rng(stable_seed(f"{cov}:{chrom}:{wn}", seed))
        decoy = diagonal_preserving_decoy(arr, rng)
        batch = np.stack([normalize_matrix(arr), normalize_matrix(decoy)], axis=0)
        x = torch.from_numpy(batch).unsqueeze(1).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            out = model(x)
            left = out["left_boundary"].float().cpu().numpy()
            right = out["right_boundary"].float().cpu().numpy()
        n = left.shape[1]
        idx = np.arange(n) + offset
        valid = idx < chrom_size
        idx = idx[valid]
        lt[idx] += left[0, valid]
        rt[idx] += right[0, valid]
        ld[idx] += left[1, valid]
        rd[idx] += right[1, valid]
        count[idx] += 1.0
    valid = count > 0
    lt[valid] /= count[valid]
    rt[valid] /= count[valid]
    ld[valid] /= count[valid]
    rd[valid] /= count[valid]
    return lt, rt, ld, rd


def predict_chromosome(model, chrom: str, data_dir: Path, cov: str, use_oe: bool, device: torch.device, peak_radius: int, fdr_alpha: float, fdr_metric: str, seed: int) -> Tuple[List[Tuple[str, int, int, float]], dict]:
    left_t, right_t, left_d, right_d = aggregate_target_decoy_tracks(model, chrom, data_dir, cov, use_oe, device, seed)
    left_candidates, left_cutoff, left_target_peaks, left_decoy_peaks = select_fdr_peaks(left_t, left_d, peak_radius, fdr_alpha, fdr_metric)
    right_candidates, right_cutoff, right_target_peaks, right_decoy_peaks = select_fdr_peaks(right_t, right_d, peak_radius, fdr_alpha, fdr_metric)
    tads = pair_boundaries(
        left_t,
        right_t,
        left_candidates,
        right_candidates,
        MIN_LEN_BINS,
        MAX_LEN_BINS,
        NMS_METRIC,
        NMS_THRESH,
        ALLOW_NESTED,
    )
    result = []
    for start_bin, end_bin, score in tads:
        start_bp = start_bin * RESOLUTION
        end_bp = (end_bin + 1) * RESOLUTION
        result.append((chrom, start_bp, end_bp, float(score)))
    stats = {
        "left_target_peaks": left_target_peaks,
        "left_decoy_peaks": left_decoy_peaks,
        "left_selected": int(len(left_candidates)),
        "left_cutoff": float(left_cutoff) if np.isfinite(left_cutoff) else "inf",
        "right_target_peaks": right_target_peaks,
        "right_decoy_peaks": right_decoy_peaks,
        "right_selected": int(len(right_candidates)),
        "right_cutoff": float(right_cutoff) if np.isfinite(right_cutoff) else "inf",
        "tads": int(len(tads)),
    }
    return result, stats


def main() -> None:
    p = argparse.ArgumentParser(description="Predict raw TAD candidates with RefHiC-style boundary FDR")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--coverages", nargs="+", default=["4000M"])
    p.add_argument("--chroms", nargs="+", default=["chr15", "chr16", "chr17"])
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--output_prefix", default="ContextTAD_structure")
    p.add_argument("--peak_radius", type=int, default=5)
    p.add_argument("--fdr_alpha", type=float, default=0.005)
    p.add_argument("--fdr_metric", choices=["prominence", "height"], default="prominence")
    args = p.parse_args()
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_args = load_model(args.ckpt, device)
    use_oe = not bool(ckpt_args.get("use_obs", False))
    print(f"Loaded {args.ckpt}")
    print(f"backbone={ckpt_args.get('backbone')} diag_radius={ckpt_args.get('diag_radius', 50)} use_oe={use_oe}")
    print(
        f"decode=target_decoy_fdr_endpoint_product metric={args.fdr_metric} "
        f"alpha={args.fdr_alpha} peak_radius={args.peak_radius}"
    )
    for cov in args.coverages:
        all_tads = []
        for chrom in args.chroms:
            tads, stats = predict_chromosome(
                model,
                chrom,
                data_dir,
                cov,
                use_oe,
                device,
                args.peak_radius,
                args.fdr_alpha,
                args.fdr_metric,
                args.seed,
            )
            all_tads.extend(tads)
            print(
                f"  {cov} {chrom}: "
                f"L={stats['left_selected']}/{stats['left_target_peaks']} cutoff={stats['left_cutoff']} decoyL={stats['left_decoy_peaks']} "
                f"R={stats['right_selected']}/{stats['right_target_peaks']} cutoff={stats['right_cutoff']} decoyR={stats['right_decoy_peaks']} "
                f"TAD={stats['tads']}",
                flush=True,
            )
        path = out_dir / f"{args.output_prefix}_{cov}.bed"
        with path.open("w") as f:
            for c, s, e, sc in sorted(all_tads):
                f.write(f"{c}\t{s}\t{e}\t{sc:.6f}\n")
        print(f"[{cov}] total={len(all_tads)} -> {path}", flush=True)


if __name__ == "__main__":
    main()
