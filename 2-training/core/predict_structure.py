#!/usr/bin/env python3
from __future__ import annotations

import random
import zlib
from typing import List, Tuple

import numpy as np
import torch
from scipy.signal import find_peaks

from model import ContextTADBackboneModel  # type: ignore
RESOLUTION = 5000
MIN_LEN_BINS = 5
MAX_LEN_BINS = 350
NMS_METRIC = "iou"
NMS_THRESH = 0.5
ALLOW_NESTED = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_seed(text: str, base: int = 42) -> int:
    return (base + zlib.adler32(text.encode("utf-8"))) % (2**32 - 1)


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
