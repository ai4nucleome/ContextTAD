#!/usr/bin/env python3
"""Minimal ContextTAD decoder for boundary-based TAD prediction.

This script intentionally keeps only the active decode path used by current
reproducible evaluation:
  1) aggregate window-level left/right predictions
  2) cross pair left-right peaks under length constraints
  3) score by logit fusion (left/right)
  4) NMS + top-k per chromosome
  5) export raw BED in end-exclusive coordinates
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent.parent
import sys
sys.path.insert(0, str(SCRIPT_DIR))

from train import INFERENCE_PROMPT, TAD_SAM3_Structure
from transformers import CLIPTokenizer, Sam3Processor

SAM3_PATH = os.environ.get("SAM3_PATH", "/home/weicai/projectnvme/TADAnno/sam3")
DEFAULT_DATA_DIR = Path("/home/weicai/projectnvme/TADAnno_final_v2/0-data/1_dp_train_infer_data")
DATA_DIR = Path(os.environ.get("TAD_DATA_DIR", str(DEFAULT_DATA_DIR)))
if not DATA_DIR.exists():
    DATA_DIR = Path("/home/weicai/projectnvme/TADAnno_final_v2/0-data/1_dp_train_infer_data")

RESOLUTION = 5000

# Fixed decode parameters (validated repro path)
DEFAULT_METHOD = "text_oe"
LORA_R = 16
THRESHOLD = 0.3
MIN_LEN_BINS = 5
MAX_LEN_BINS = 350
TOP_K_PER_CHROM = 5000
NMS_METRIC = "iou"
NMS_THRESH = 0.5
ALLOW_NESTED = True


W_LEFT = 1.0
W_RIGHT = 1.0

PROMPT_PRESETS = {
    "default": INFERENCE_PROMPT,
    "p1": "topologically associating domain",
    "p2": "a dog in the park",
    "p3": "non-topologically associating domain in Hi-C contact map",
}


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_logit(x: float, eps: float = 1e-6) -> float:
    x = float(np.clip(x, eps, 1.0 - eps))
    return float(np.log(x / (1.0 - x)))


def load_model(ckpt_path: str, method: str = "auto", force_disable_text: bool = False):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args") if isinstance(ckpt, dict) else None
    if not isinstance(ckpt_args, dict):
        ckpt_args = {}

    prompt_policy = str(ckpt_args.get("prompt_policy", "fixed"))
    if prompt_policy != "fixed":
        raise ValueError(
            f"Checkpoint prompt_policy='{prompt_policy}' is not supported in inference. "
            "Use fixed prompt policy for train/infer consistency."
        )

    ckpt_method = str(ckpt_args.get("method", DEFAULT_METHOD))
    model_method = ckpt_method if method == "auto" else method

    use_oe = model_method == "text_oe"
    use_text = model_method in ("text_obs", "text_oe")
    if bool(ckpt_args.get("disable_text_branch", False)) or force_disable_text:
        use_text = False

    use_tofe = not bool(ckpt_args.get("disable_tofe", False))
    tofe_mix = float(ckpt_args.get("tofe_mix", 1.0))

    model = TAD_SAM3_Structure(
        method=model_method,
        lora_r=LORA_R,
        use_tofe=use_tofe,
        tofe_mix=tofe_mix,
        force_use_text=use_text,
        force_use_oe=use_oe,
    )

    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    info = {
        "method": model_method,
        "use_oe": use_oe,
        "use_text": use_text,
        "use_tofe": use_tofe,
    }
    return model.cuda(), info


def pair_boundaries(
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    threshold: float,
    min_len_bins: int,
    max_len_bins: int,
    top_k: int,
    w_left: float,
    w_right: float,
    nms_metric: str,
    nms_thresh: float,
    allow_nested: bool,
) -> List[Tuple[int, int, float]]:
    left_candidates = np.where(left_scores > threshold)[0]
    right_candidates = np.where(right_scores > threshold)[0]
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        return []

    candidates: List[Tuple[int, int, float]] = []
    for l in left_candidates:
        for r in right_candidates:
            length = r - l + 1
            if l >= r or length < min_len_bins or length > max_len_bins:
                continue
            score_logit = (
                w_left * _safe_logit(left_scores[l])
                + w_right * _safe_logit(right_scores[r])
            )
            score = 1.0 / (1.0 + np.exp(-score_logit))
            candidates.append((int(l), int(r), float(score)))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[2], reverse=True)

    selected: List[Tuple[int, int, float]] = []
    for l, r, sc in candidates:
        overlap = False
        for sl, sr, _ in selected:
            inter = min(r, sr) - max(l, sl) + 1
            if inter <= 0:
                continue

            if allow_nested:
                contained = (l >= sl and r <= sr) or (sl >= l and sr <= r)
                if contained:
                    continue

            len1 = r - l + 1
            len2 = sr - sl + 1
            if nms_metric == "iou":
                union = len1 + len2 - inter
                suppress_score = inter / max(union, 1)
            else:
                suppress_score = inter / max(min(len1, len2), 1)

            if suppress_score > nms_thresh:
                overlap = True
                break

        if not overlap:
            selected.append((l, r, sc))
            if len(selected) >= top_k:
                break

    return selected


def predict_chromosome(
    model,
    chrom: str,
    data_dir: Path,
    cov: str,
    processor,
    tokenizer,
    use_oe: bool,
    use_text: bool,
    prompt_text: str,
) -> List[Tuple[str, int, int, float]]:
    cov_dir = data_dir / cov / chrom
    if not cov_dir.exists():
        return []

    with open(data_dir / "window_list.json") as f:
        wl = json.load(f)
    test_windows = wl["windows"]["test"]
    win_names = sorted([w for w in test_windows if w.startswith(chrom + "_")])
    if not win_names:
        return []

    text_ids = text_mask = None
    if use_text and tokenizer is not None:
        tok = tokenizer(
            prompt_text,
            return_tensors="pt",
            padding="max_length",
            max_length=32,
            truncation=True,
        )
        text_ids = tok["input_ids"].cuda()
        text_mask = tok["attention_mask"].cuda()

    chrom_size = max(int(w.split("_")[-1]) for w in win_names) + 400

    left_strength = np.zeros(chrom_size, dtype=np.float32)
    right_strength = np.zeros(chrom_size, dtype=np.float32)
    count = np.zeros(chrom_size, dtype=np.float32)

    for wn in win_names:
        offset = int(wn.split("_")[-1])
        npy_path = cov_dir / f"{wn}.npy"
        if not npy_path.exists():
            continue

        mat = np.load(npy_path)
        obs = mat[1] if use_oe else mat[0]

        if use_oe:
            lm = np.log1p(obs.astype(np.float32))
        else:
            lm = np.log10(obs.astype(np.float32) + 1)
        mx = max(lm.max(), 1e-6)
        inv = (mx - lm) / mx
        oe_values = np.clip(lm / mx, 0.0, 1.0).astype(np.float32)
        rgb = np.stack([np.ones_like(lm) * 255, inv * 255, inv * 255], axis=-1).astype(np.uint8)
        pv = processor(images=Image.fromarray(rgb, "RGB"), return_tensors="pt")["pixel_values"].cuda()

        fwd_kwargs = {
            "pixel_values": pv,
            "oe_values": torch.from_numpy(oe_values).unsqueeze(0).unsqueeze(0).cuda(),
        }
        if use_text and text_ids is not None:
            fwd_kwargs["text_input_ids"] = text_ids
            if text_mask is not None:
                fwd_kwargs["text_attention_mask"] = text_mask

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(**fwd_kwargs)
            left_pred = outputs["left_boundary"].float().cpu().numpy().squeeze()
            right_pred = outputs["right_boundary"].float().cpu().numpy().squeeze()

        for i in range(len(left_pred)):
            gi = offset + i
            if gi < chrom_size:
                left_strength[gi] += left_pred[i]
                right_strength[gi] += right_pred[i]
                count[gi] += 1.0

    valid = count > 0
    left_strength[valid] /= count[valid]
    right_strength[valid] /= count[valid]

    tads = pair_boundaries(
        left_scores=left_strength,
        right_scores=right_strength,
        threshold=THRESHOLD,
        min_len_bins=MIN_LEN_BINS,
        max_len_bins=MAX_LEN_BINS,
        top_k=TOP_K_PER_CHROM,
        w_left=W_LEFT,
        w_right=W_RIGHT,
        nms_metric=NMS_METRIC,
        nms_thresh=NMS_THRESH,
        allow_nested=ALLOW_NESTED,
    )

    result: List[Tuple[str, int, int, float]] = []
    for start_bin, end_bin, score in tads:
        start_bp = start_bin * RESOLUTION
        # Keep end-exclusive raw BED semantics. Postprocess converts to right-boundary coordinate.
        end_bp = (end_bin + 1) * RESOLUTION
        result.append((chrom, start_bp, end_bp, float(score)))

    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Minimal ContextTAD decoder")
    p.add_argument("--ckpt", required=True, help="Path to trained model checkpoint")
    p.add_argument("--output_dir", required=True, help="Output directory for predictions")
    p.add_argument("--coverages", nargs="+", default=["4000M"], help="Coverage levels to predict")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--chroms", nargs="+", default=["chr15", "chr16", "chr17"], help="Chromosomes to predict")
    p.add_argument(
        "--method",
        default="auto",
        choices=["auto", "text_obs", "text_oe"],
        help="Model method. auto: infer from checkpoint args.",
    )
    p.add_argument(
        "--disable_text_branch",
        action="store_true",
        help="Force-disable text branch at inference regardless of checkpoint args.",
    )
    p.add_argument(
        "--output_prefix",
        default="ContextTAD_structure",
        help="Raw BED output filename prefix; final file is <prefix>_<coverage>.bed",
    )
    p.add_argument(
        "--prompt_id",
        default="default",
        choices=["default", "p1", "p2", "p3"],
        help="Inference prompt preset.",
    )
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    set_reproducible_seed(args.seed)

    print(f"Loading model from {args.ckpt}")
    model, model_info = load_model(
        args.ckpt,
        method=args.method,
        force_disable_text=args.disable_text_branch,
    )

    processor = Sam3Processor.from_pretrained(SAM3_PATH)
    tokenizer = CLIPTokenizer.from_pretrained(SAM3_PATH) if model_info["use_text"] else None

    print(f"Predicting for chromosomes: {args.chroms}")
    print(f"Coverage levels: {args.coverages}")
    print(
        "Model config: "
        f"method={model_info['method']} use_oe={model_info['use_oe']} "
        f"use_text={model_info['use_text']} use_tofe={model_info['use_tofe']}"
    )
    print(f"Threshold: {THRESHOLD}, Min length: {MIN_LEN_BINS} bins")
    print(f"Seed: {args.seed}")
    print(f"Fusion weights: left={W_LEFT}, right={W_RIGHT}")
    print(f"Top-K per chromosome: {TOP_K_PER_CHROM}")
    print(f"Suppression: metric={NMS_METRIC}, thresh={NMS_THRESH}, allow_nested={ALLOW_NESTED}")
    prompt_text = PROMPT_PRESETS.get(args.prompt_id, INFERENCE_PROMPT)
    print(f"Prompt preset: {args.prompt_id}")
    print(f"Prompt text: {prompt_text}")
    print("Cross-attn text mask enabled: True")

    for cov in args.coverages:
        all_tads: List[Tuple[str, int, int, float]] = []
        for ch in args.chroms:
            tads = predict_chromosome(
                model=model,
                chrom=ch,
                data_dir=DATA_DIR,
                cov=cov,
                processor=processor,
                tokenizer=tokenizer,
                use_oe=model_info["use_oe"],
                use_text=model_info["use_text"],
                prompt_text=prompt_text,
            )
            all_tads.extend(tads)
            print(f"  {ch}: {len(tads)} TADs")

        bed_path = out / f"{args.output_prefix}_{cov}.bed"
        with open(bed_path, "w") as f:
            for c, s, e, sc in sorted(all_tads):
                f.write(f"{c}\t{s}\t{e}\t{sc:.6f}\n")
        print(f"[{cov}] Total: {len(all_tads)} TADs -> {bed_path}")

    print("\nPrediction complete!")


if __name__ == "__main__":
    main()
