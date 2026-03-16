#!/usr/bin/env python3
"""
Structure-aware TAD training entry.

Core path:
- Input: obs/OE window matrices
- Backbone: SAM3 vision encoder + text-guided fusion
- Heads: left boundary, right boundary, presence, nesting depth, and pair matrix
- Loss: weighted combination configured by command-line arguments

Example:
  conda run -n 3dgenome python scripts/train_structure.py --method text_oe --epochs 10
"""

import os
import sys
import argparse
import json
import time
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from PIL import Image
from tqdm import tqdm

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from transformers import Sam3Model, Sam3Processor, CLIPTokenizer
from peft import get_peft_model, LoraConfig

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent.parent
DEFAULT_DATA_DIR = Path("/home/weicai/projectnvme/TADAnno_final/0-data/1_dp_train_infer_data")
DATA_DIR = Path(os.environ.get("TAD_DATA_DIR", str(DEFAULT_DATA_DIR)))
SAM3_PATH = os.environ.get("SAM3_PATH", "/home/weicai/projectnvme/TADAnno/sam3")

if not DATA_DIR.exists():
    DATA_DIR = Path("/home/weicai/projectnvme/TADAnno_final/0-data/1_dp_train_infer_data")

sys.path.insert(0, str(SCRIPT_DIR))
from losses import TADStructureLoss
from tofe import TOFE
from prompt_policy import FIXED_PROMPT

RESOLUTION = 5000
DEFAULT_COVERAGES = ["4000M", "2000M", "1000M", "500M", "250M", "125M", "62_5M"]

# TAD description helper for text prompt augmentation.

def generate_tad_descriptions(tads_npy, boundary_mask):
    """Generate short textual descriptions from GT TAD annotations."""
    valid_tads = tads_npy[tads_npy[:, 0] >= 0]
    n_tads = len(valid_tads)
    n_boundaries = int(boundary_mask.sum())

    if n_tads > 0:
        sizes = valid_tads[:, 1] - valid_tads[:, 0] + 1
        mean_size = int(sizes.mean())
        min_size = int(sizes.min())
        max_size = int(sizes.max())
    else:
        mean_size = min_size = max_size = 0

    descriptions = [
        "topologically associating domain boundary in Hi-C map",
        "chromatin domain boundary with enriched contacts",
        "TAD boundary separating self-interacting genomic regions",
        f"Hi-C region with {n_tads} topological domains and {n_boundaries} boundaries",
        f"chromatin with domains from {min_size} to {max_size} bins" if n_tads > 0 else "sparse genomic region with few domains",
        "insulated genomic neighborhood boundary marked by CTCF",
        "boundary between self-interacting chromatin domains",
    ]
    return descriptions


INFERENCE_PROMPT = FIXED_PROMPT


# Text-guided Cross-Attention Module

class TextGuidedCrossAttention(nn.Module):
    """ text features : Q=visual, KV=text"""

    def __init__(self, vis_dim=128, text_dim=1024, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = vis_dim // n_heads

        self.q_proj = nn.Linear(vis_dim, vis_dim)
        self.k_proj = nn.Linear(text_dim, vis_dim)
        self.v_proj = nn.Linear(text_dim, vis_dim)
        self.out_proj = nn.Linear(vis_dim, vis_dim)
        self.norm = nn.LayerNorm(vis_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, vis_feat, text_feat, text_attention_mask=None):
        """
        vis_feat: [B, N, vis_dim]
        text_feat: [B, T, text_dim]
        text_attention_mask: [B, T], 1 valid, 0 padding
        """
        B, N, D = vis_feat.shape
        residual = vis_feat

        q = self.q_proj(vis_feat).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(text_feat).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(text_feat).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if text_attention_mask is not None:
            tok_mask = text_attention_mask.to(device=attn.device, dtype=torch.bool)[:, None, None, :]
            attn = attn.masked_fill(~tok_mask, -1e4)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        return self.norm(residual + out)


# Diagonal Projection (2D  1D)

class DiagonalProjection1D(nn.Module):
    def __init__(self, in_ch, hidden=128, out_size=400, dw=5):
        super().__init__()
        self.out_size = out_size
        self.dw = dw
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU()
        )
        self.fc = nn.Linear(hidden * dw, hidden)

    def forward(self, feat2d):
        if feat2d.shape[-1] != self.out_size:
            feat2d = F.interpolate(feat2d, (self.out_size, self.out_size),
                                   mode="bilinear", align_corners=False)
        x = self.conv(feat2d)
        B, C, H, W = x.shape
        diags = []
        hw = self.dw // 2
        for off in range(-hw, hw + 1):
            d = torch.diagonal(x, offset=off, dim1=2, dim2=3)
            if off != 0:
                ps = abs(off)
                d = F.pad(d, (ps, 0) if off < 0 else (0, ps))
            diags.append(d)
        diags = torch.cat(diags, dim=1).transpose(1, 2)  # [B, N, C*dw]
        return self.fc(diags)  # [B, N, hidden]


# Structure-aware Dual-Boundary Decoder

class StructureAwareDecoder(nn.Module):
    """
    
    
    :
    - left_boundary: [B, N] 
    - right_boundary: [B, N] 
    - presence: [B, 1] TAD
    - nesting_depth: [B, N] 
    """

    def __init__(self, hidden=128, out_size=400):
        super().__init__()
        self.hidden = hidden
        self.out_size = out_size
        
        self.shared = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        
        self.left_head = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, 3, padding=1),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Conv1d(hidden // 2, 1, 1),
        )
        
        self.right_head = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, 3, padding=1),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Conv1d(hidden // 2, 1, 1),
        )

        #  pair  ( left-right )
        pair_dim = 64
        self.pair_left_proj = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, 1),
            nn.GELU(),
            nn.Conv1d(hidden // 2, pair_dim, 1),
        )
        self.pair_right_proj = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, 1),
            nn.GELU(),
            nn.Conv1d(hidden // 2, pair_dim, 1),
        )
        
        # Presence ( + MLP)
        self.presence_pool = nn.AdaptiveAvgPool1d(1)
        self.presence_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )
        
        # Nesting Depth ()
        self.depth_head = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, 3, padding=1),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Conv1d(hidden // 2, 1, 1),
        )

    def forward(self, diag_feat):
        """
        diag_feat: [B, N, hidden] from diagonal projection
        
        Returns:
            dict with keys: left_boundary, right_boundary, presence, nesting_depth
        """
        # [B, N, hidden]  [B, hidden, N]
        x = diag_feat.transpose(1, 2)
        
        shared_feat = self.shared(x)  # [B, hidden, N]
        
        left_logits = self.left_head(shared_feat).squeeze(1)    # [B, N]
        right_logits = self.right_head(shared_feat).squeeze(1)  # [B, N]
        
        # Presence ()
        pooled = self.presence_pool(shared_feat).squeeze(-1)    # [B, hidden]
        presence = self.presence_head(pooled)                   # [B, 1]
        
        # Nesting Depth
        depth = self.depth_head(shared_feat).squeeze(1)         # [B, N]

        # Pair compatibility map: [B, N, N]
        pair_left = F.normalize(self.pair_left_proj(shared_feat).transpose(1, 2), dim=-1)
        pair_right = F.normalize(self.pair_right_proj(shared_feat).transpose(1, 2), dim=-1)
        pair_logits = torch.einsum("bid,bjd->bij", pair_left, pair_right)
        pair_map = torch.sigmoid(pair_logits)

        return {
            'left_boundary': torch.sigmoid(left_logits),
            'right_boundary': torch.sigmoid(right_logits),
            'presence': presence,
            'nesting_depth': F.relu(depth),  # 
            'pair_map': pair_map,
        }


# Main Model: TAD_SAM3_Structure

class TAD_SAM3_Structure(nn.Module):
    """
    Structure-aware TAD detection model
    
    Architecture:
    1. SAM3 ViT backbone (frozen + LoRA)
    2. FPN feature projection
    3. Diagonal projection (2D  1D)
    4. Optional: Text-guided cross-attention
    5. Structure-aware decoder (left + right boundary)
    """

    def __init__(
        self,
        method="text_oe",
        sam3_path=SAM3_PATH,
        lora_r=16,
        hidden=128,
        use_tofe=True,
        tofe_mix=1.0,
        force_use_text: Optional[bool] = None,
        force_use_oe: Optional[bool] = None,
    ):
        super().__init__()
        self.method = method
        base_use_text = method in ("text_obs", "text_oe", "combo")
        base_use_oe = method in ("text_oe", "combo")
        self.use_text = bool(base_use_text if force_use_text is None else force_use_text)
        self.use_oe = bool(base_use_oe if force_use_oe is None else force_use_oe)
        self.use_tofe = use_tofe
        self.tofe_mix = float(max(min(tofe_mix, 1.0), 0.0))
        
        # SAM3 backbone
        self.sam3 = Sam3Model.from_pretrained(sam3_path)
        for p in self.sam3.parameters():
            p.requires_grad = False

        self.sam3 = get_peft_model(self.sam3, LoraConfig(
            r=lora_r, lora_alpha=lora_r * 2,
            target_modules=r"vision_encoder\..*\.(q_proj|v_proj)",
            lora_dropout=0.05, bias="none",
        ))

        # FPN  hidden
        self.feature_proj = nn.Sequential(
            nn.Conv2d(256 * 3, hidden, 1),
            nn.BatchNorm2d(hidden), 
            nn.GELU()
        )

        # Diagonal projection: 2D  1D
        self.diag_proj = DiagonalProjection1D(hidden, hidden, 400)

        # Text cross-attention (optional)
        if self.use_text:
            self.text_cross_attn = TextGuidedCrossAttention(
                vis_dim=hidden, text_dim=1024, n_heads=4)

        # Structure-aware decoder
        self.decoder = StructureAwareDecoder(hidden, 400)

        if self.use_tofe:
            self.tofe = TOFE(in_channels=1, hidden=32)

    def encode_text(self, text_input_ids, text_attention_mask):
        with torch.no_grad():
            text_out = self.sam3.base_model.model.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
            )
        return text_out.last_hidden_state  # [B, seq_len, 1024]

    def forward(self, pixel_values, text_input_ids=None, text_attention_mask=None, oe_values=None):
        if self.use_tofe and oe_values is not None:
            enhanced = self.tofe(oe_values.float())
            enhanced = F.interpolate(
                enhanced,
                size=(pixel_values.shape[-2], pixel_values.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            enhanced = enhanced * 2.0 - 1.0
            pixel_values = (1.0 - self.tofe_mix) * pixel_values + self.tofe_mix * enhanced

        # Vision encoder  FPN
        vo = self.sam3.base_model.model.vision_encoder(pixel_values)
        fpn = vo.fpn_hidden_states
        target_size = (400, 400)
        f0 = F.interpolate(fpn[0].float(), target_size, mode="bilinear", align_corners=False)
        f1 = F.interpolate(fpn[1].float(), target_size, mode="bilinear", align_corners=False)
        f2 = F.interpolate(fpn[2].float(), target_size, mode="bilinear", align_corners=False)
        feat = self.feature_proj(torch.cat([f0, f1, f2], dim=1))  # [B, hidden, 400, 400]

        # Diagonal projection
        diag_feat = self.diag_proj(feat)  # [B, 400, hidden]

        # Text cross-attention
        if self.use_text and text_input_ids is not None:
            text_feat = self.encode_text(text_input_ids, text_attention_mask)
            diag_feat = self.text_cross_attn(diag_feat, text_feat.float(), text_attention_mask)

        # Decode
        outputs = self.decoder(diag_feat)
        
        return outputs

    def get_param_groups(self, lr):
        """"""
        lora_p = [p for n, p in self.named_parameters() if p.requires_grad and "lora" in n]
        head_p = [p for n, p in self.named_parameters() if p.requires_grad and "lora" not in n]
        return [
            {"params": lora_p, "lr": lr * 0.1, "name": "lora"},
            {"params": head_p, "lr": lr, "name": "head"}
        ]


# Dataset

class TADStructureDataset(Dataset):
    """
    Structure-aware TAD Dataset
    
    :
    - tads.npy  TAD list ( pair loss)
    - left/right boundary masks ( TAD )
    """

    def __init__(
        self,
        data_dir,
        split="train",
        use_oe=False,
        use_text=False,
        use_soft_boundary_targets=False,
        boundary_sigma=1.0,
        boundary_count_clip=3.0,
        coverages=None,
        sampling_mode=None,
        fixed_cov="4000M",
    ):
        self.data_dir = Path(data_dir)
        with open(self.data_dir / "window_list.json") as f:
            wl = json.load(f)
        self.windows = wl["windows"][split]
        self.covs = coverages or DEFAULT_COVERAGES
        self.split = split
        if sampling_mode is None:
            sampling_mode = "random_single" if split == "train" else "fixed_single"
        self.sampling_mode = sampling_mode
        self.fixed_cov = fixed_cov
        self.use_oe = use_oe
        self.use_text = use_text
        self.use_soft_boundary_targets = use_soft_boundary_targets
        self.boundary_sigma = max(float(boundary_sigma), 0.0)
        self.boundary_count_clip = max(float(boundary_count_clip), 1.0)
        self.processor = Sam3Processor.from_pretrained(SAM3_PATH)
        if use_text:
            self.tokenizer = CLIPTokenizer.from_pretrained(SAM3_PATH)

        if self.sampling_mode == "all_pairs":
            self.index_items = [(wn, cov) for wn in self.windows for cov in self.covs]
        else:
            self.index_items = list(self.windows)

    def __len__(self):
        return len(self.index_items)

    def _gaussian_smooth(self, arr):
        if self.boundary_sigma <= 0:
            return arr.astype(np.float32)
        radius = max(1, int(math.ceil(self.boundary_sigma * 3)))
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel = np.exp(-(x * x) / (2 * self.boundary_sigma * self.boundary_sigma))
        kernel = kernel / max(kernel.sum(), 1e-6)
        return np.convolve(arr, kernel, mode="same").astype(np.float32)

    def _extract_boundary_masks(self, tads, window_size=400):
        left_count = np.zeros(window_size, dtype=np.float32)
        right_count = np.zeros(window_size, dtype=np.float32)

        valid_tads = tads[tads[:, 0] >= 0]
        for start, end in valid_tads:
            start, end = int(start), int(end)
            if start < window_size:
                left_count[start] += 1.0
            if end - 1 < window_size and end > 0:
                right_count[end - 1] += 1.0

        if self.use_soft_boundary_targets:
            left_target = np.clip(left_count / self.boundary_count_clip, 0.0, 1.0)
            right_target = np.clip(right_count / self.boundary_count_clip, 0.0, 1.0)
            left_target = np.clip(self._gaussian_smooth(left_target), 0.0, 1.0)
            right_target = np.clip(self._gaussian_smooth(right_target), 0.0, 1.0)
        else:
            left_target = (left_count > 0).astype(np.float32)
            right_target = (right_count > 0).astype(np.float32)

        return left_target, right_target, float(left_count.sum()), float(right_count.sum())

    def __getitem__(self, idx):
        if self.sampling_mode == "all_pairs":
            wn, cov = self.index_items[idx]
        else:
            wn = self.index_items[idx]
            if self.sampling_mode == "random_single":
                cov = random.choice(self.covs)
            else:
                cov = self.fixed_cov

        ch = wn[:wn.rfind("_")]

        mat = np.load(self.data_dir / cov / ch / f"{wn}.npy")
        obs = mat[1] if self.use_oe else mat[0]
        labels = np.load(self.data_dir / "labels" / ch / f"{wn}_labels.npy")
        
        # Load TADs for structure loss
        tads = np.load(self.data_dir / "labels" / ch / f"{wn}_tads.npy")

        # Hi-C  RGB
        lm = np.log1p(obs.astype(np.float32))
        # if self.use_oe:
        #     lm = np.log1p(obs.astype(np.float32))
        # else:
        #     lm = np.log10(obs.astype(np.float32) + 1)
        mx = max(lm.max(), 1e-6)
        inv = (mx - lm) / mx
        oe_values = np.clip(lm / mx, 0.0, 1.0).astype(np.float32)
        rgb = np.stack([np.ones_like(lm) * 255, inv * 255, inv * 255], axis=-1).astype(np.uint8)
        pv = self.processor(images=Image.fromarray(rgb, "RGB"),
                            return_tensors="pt")["pixel_values"].squeeze(0)

        # Extract boundary masks from TADs
        left_mask, right_mask, left_count_gt, right_count_gt = self._extract_boundary_masks(tads)
        
        # Presence: TAD
        presence_gt = float(len(tads[tads[:, 0] >= 0]) > 0)
        
        result = {
            "pixel_values": pv,
            "boundary_score": torch.from_numpy(labels[:, 0].astype(np.float32)),
            "boundary_mask": torch.from_numpy(labels[:, 1].astype(np.float32)),
            "nesting_depth": torch.from_numpy(labels[:, 2].astype(np.float32)),
            "left_mask": torch.from_numpy(left_mask),
            "right_mask": torch.from_numpy(right_mask),
            "left_count_gt": torch.tensor([left_count_gt], dtype=torch.float32),
            "right_count_gt": torch.tensor([right_count_gt], dtype=torch.float32),
            "presence_gt": torch.tensor([presence_gt], dtype=torch.float32),
            "tads": torch.from_numpy(tads.astype(np.int32)),  # For pair loss
            "oe_values": torch.from_numpy(oe_values).unsqueeze(0),
        }

        # Text
        if self.use_text:
            tok = self.tokenizer(INFERENCE_PROMPT, return_tensors="pt", padding="max_length",
                                 max_length=32, truncation=True)
            result["text_input_ids"] = tok["input_ids"].squeeze(0)
            result["text_attention_mask"] = tok["attention_mask"].squeeze(0)

        return result


def collate_fn(batch):
    """Custom collate function to handle TAD lists of different sizes"""
    # Standard fields
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    boundary_score = torch.stack([b["boundary_score"] for b in batch])
    boundary_mask = torch.stack([b["boundary_mask"] for b in batch])
    nesting_depth = torch.stack([b["nesting_depth"] for b in batch])
    left_mask = torch.stack([b["left_mask"] for b in batch])
    right_mask = torch.stack([b["right_mask"] for b in batch])
    left_count_gt = torch.stack([b["left_count_gt"] for b in batch])
    right_count_gt = torch.stack([b["right_count_gt"] for b in batch])
    presence_gt = torch.stack([b["presence_gt"] for b in batch])
    
    # TADs (variable length)
    tads_list = [b["tads"] for b in batch]
    
    result = {
        "pixel_values": pixel_values,
        "boundary_score": boundary_score,
        "boundary_mask": boundary_mask,
        "nesting_depth": nesting_depth,
        "left_mask": left_mask,
        "right_mask": right_mask,
        "left_count_gt": left_count_gt,
        "right_count_gt": right_count_gt,
        "presence_gt": presence_gt,
        "tads_list": tads_list,
        "oe_values": torch.stack([b["oe_values"] for b in batch]),
    }
    
    # Optional text fields
    if "text_input_ids" in batch[0]:
        result["text_input_ids"] = torch.stack([b["text_input_ids"] for b in batch])
        result["text_attention_mask"] = torch.stack([b["text_attention_mask"] for b in batch])
    
    return result


# Training

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="text_oe",
                   choices=["text_obs", "text_oe", "enhanced", "combo"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    
    # Active loss weights (current training path)
    p.add_argument("--pair_weight", type=float, default=2.0)
    p.add_argument("--count_weight", type=float, default=0.0)
    p.add_argument("--pair_hard_window", type=int, default=8)
    p.add_argument("--disable_pair_hard_negatives", action="store_true")

    p.add_argument("--use_soft_boundary_targets", action="store_true")
    p.add_argument("--boundary_sigma", type=float, default=1.0)
    p.add_argument("--boundary_count_clip", type=float, default=3.0)
    p.add_argument("--disable_tofe", action="store_true")
    p.add_argument("--disable_text_branch", action="store_true",
                   help="Disable text cross-attention branch while keeping other settings unchanged.")
    p.add_argument("--tofe_mix", type=float, default=1.0)

    p.add_argument("--init_ckpt", type=str, default=None)
    p.add_argument(
        "--train_sampling",
        choices=["random_single", "all_pairs"],
        default="random_single",
        help="How train split samples coverages: random_single (legacy) or all_pairs (window x coverage).",
    )
    p.add_argument(
        "--val_sampling",
        choices=["fixed_single", "random_single", "all_pairs"],
        default="random_single",
        help="How val split samples coverages: fixed_single / random_single / all_pairs.",
    )
    p.add_argument(
        "--val_fixed_cov",
        default="4000M",
        help="Coverage used when --val_sampling=fixed_single.",
    )
    
    # Output
    p.add_argument("--output_dir", type=str, default=None,
                   help="Output directory (default: outputs/structure_{method})")
    p.add_argument(
        "--save_every_epoch",
        action="store_true",
        help="Save one checkpoint per epoch under <output_dir>/checkpoints.",
    )
    p.add_argument(
        "--stop_after_epoch",
        type=int,
        default=0,
        help="Stop training after this epoch (1-based). 0 means run all epochs.",
    )
    
    return p.parse_args()


def cosine_lr(step, total, warmup):
    if step < warmup:
        return step / max(warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))


def main():
    args = parse_args()
    method = args.method
    use_oe = method in ("text_oe", "combo")
    use_text = method in ("text_obs", "text_oe", "combo")
    if args.disable_text_branch:
        use_text = False

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = PROJECT / "outputs" / f"structure_{method}"

    ddp_kw = DistributedDataParallelKwargs(find_unused_parameters=True)
    acc = Accelerator(mixed_precision="fp16", log_with="tensorboard",
                      project_dir=str(out_dir), kwargs_handlers=[ddp_kw])
    set_seed(args.seed)
    if acc.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = out_dir / "checkpoints"
        if args.save_every_epoch:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Save args
        with open(out_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    acc.wait_for_everyone()

    # Data
    train_ds = TADStructureDataset(
        DATA_DIR,
        "train",
        use_oe=use_oe,
        use_text=use_text,
        use_soft_boundary_targets=args.use_soft_boundary_targets,
        boundary_sigma=args.boundary_sigma,
        boundary_count_clip=args.boundary_count_clip,
        coverages=DEFAULT_COVERAGES,
        sampling_mode=args.train_sampling,
        fixed_cov="4000M",
    )
    val_ds = TADStructureDataset(
        DATA_DIR,
        "val",
        use_oe=use_oe,
        use_text=use_text,
        use_soft_boundary_targets=args.use_soft_boundary_targets,
        boundary_sigma=args.boundary_sigma,
        boundary_count_clip=args.boundary_count_clip,
        coverages=DEFAULT_COVERAGES,
        sampling_mode=args.val_sampling,
        fixed_cov=args.val_fixed_cov,
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True,
                          collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # Model
    acc.print(
        f"Building structure-aware model: method={method} use_oe={use_oe} "
        f"use_text={use_text} disable_tofe={args.disable_tofe} disable_text_branch={args.disable_text_branch}"
    )
    model = TAD_SAM3_Structure(
        method=method,
        lora_r=args.lora_r,
        use_tofe=not args.disable_tofe,
        tofe_mix=args.tofe_mix,
        force_use_text=use_text,
        force_use_oe=use_oe,
    )
    if args.init_ckpt:
        ckpt = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        acc.print(
            f"Loaded init checkpoint: {args.init_ckpt} | missing={len(missing)} unexpected={len(unexpected)}"
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    acc.print(f"Trainable parameters: {trainable / 1e6:.1f}M")

    # Loss
    criterion = TADStructureLoss(
        pair_weight=args.pair_weight,
        count_weight=args.count_weight,
        pair_hard_window=args.pair_hard_window,
        use_pair_hard_negatives=not args.disable_pair_hard_negatives,
    )
    acc.print(
        "Loss config: "
        f"pair={args.pair_weight}, count={args.count_weight}, "
        f"pair_hard_window={args.pair_hard_window}, "
        f"pair_hard_negatives={not args.disable_pair_hard_negatives}"
    )

    # history.json should store weighted (effective) component values
    metric_weights = {
        "pair": float(args.pair_weight),
        "count": float(args.count_weight),
    }

    def to_weighted_metrics(raw_metrics):
        weighted = {}
        for k, v in raw_metrics.items():
            fv = float(v)
            if k == "loss":
                weighted[k] = fv
            elif k in metric_weights:
                weighted[k] = fv * metric_weights[k]
            else:
                weighted[k] = fv
        return weighted

    # Optimizer
    pg = model.get_param_groups(args.lr)
    opt = AdamW(pg, weight_decay=0.01)

    spe = len(train_dl)
    total_steps = spe * args.epochs
    warmup = spe * 2
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_lr(s, total_steps, warmup))

    model, opt, train_dl, val_dl, sched = acc.prepare(model, opt, train_dl, val_dl, sched)
    acc.print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Epochs: {args.epochs}")
    acc.print(f"Sampling mode: train={args.train_sampling}, val={args.val_sampling}, val_fixed_cov={args.val_fixed_cov}")
    if args.train_sampling == "random_single":
        acc.print(
            "[note] train random_single means each epoch samples one random coverage per window "
            "(NOT full window x 7 coverage traversal)."
        )

    if acc.is_main_process:
        acc.init_trackers(f"structure_{method}",
                          config={"method": method, "lr": args.lr, "lora_r": args.lora_r,
                                  "use_oe": use_oe, "use_text": use_text,
                                  **{k: v for k, v in vars(args).items() if "weight" in k}})

    best_val = float("inf")
    history = []

    for epoch in range(args.epochs):
        # Training
        model.train()
        t_total, t_n = {}, 0
        pbar = tqdm(train_dl, desc=f"E{epoch + 1}", leave=False, disable=not acc.is_main_process)
        
        for batch in pbar:
            pv = batch["pixel_values"]
            
            # Prepare model inputs
            fwd_kwargs = {"pixel_values": pv}
            if use_text:
                fwd_kwargs["text_input_ids"] = batch["text_input_ids"]
                fwd_kwargs["text_attention_mask"] = batch["text_attention_mask"]
            fwd_kwargs["oe_values"] = batch["oe_values"]

            # Forward
            outputs = model(**fwd_kwargs)
            
            # Prepare targets
            targets = {
                "left_mask": batch["left_mask"],
                "right_mask": batch["right_mask"],
                "left_count_gt": batch["left_count_gt"],
                "right_count_gt": batch["right_count_gt"],
                "presence_gt": batch["presence_gt"],
                "depth_gt": batch["nesting_depth"],
                "boundary_mask": batch["boundary_mask"],
            }
            
            # Compute loss
            losses = criterion(outputs, targets, tads_list=batch["tads_list"])

            # Backward
            acc.backward(losses["loss"])
            if acc.sync_gradients:
                acc.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()

            # Logging
            bs = pv.size(0)
            t_n += bs
            for k, v in losses.items():
                v_scalar = float(v.item()) if hasattr(v, "item") else float(v)
                t_total[k] = t_total.get(k, 0) + v_scalar * bs
            loss_scalar = float(losses["loss"].item()) if hasattr(losses["loss"], "item") else float(losses["loss"])
            pbar.set_postfix(loss=f"{loss_scalar:.4f}")

        tm = {k: v / t_n for k, v in t_total.items()}

        # Validation
        model.eval()
        v_total, v_n = {}, 0
        with torch.no_grad():
            for batch in tqdm(val_dl, desc="val", leave=False, disable=not acc.is_main_process):
                fwd_kwargs = {"pixel_values": batch["pixel_values"]}
                if use_text:
                    fwd_kwargs["text_input_ids"] = batch["text_input_ids"]
                    fwd_kwargs["text_attention_mask"] = batch["text_attention_mask"]
                fwd_kwargs["oe_values"] = batch["oe_values"]
                
                outputs = model(**fwd_kwargs)
                
                targets = {
                    "left_mask": batch["left_mask"],
                    "right_mask": batch["right_mask"],
                    "left_count_gt": batch["left_count_gt"],
                    "right_count_gt": batch["right_count_gt"],
                    "presence_gt": batch["presence_gt"],
                    "depth_gt": batch["nesting_depth"],
                    "boundary_mask": batch["boundary_mask"],
                }
                
                losses = criterion(outputs, targets, tads_list=batch["tads_list"])
                
                bs = batch["pixel_values"].size(0)
                v_n += bs
                for k, v in losses.items():
                    v_scalar = float(v.item()) if hasattr(v, "item") else float(v)
                    v_total[k] = v_total.get(k, 0) + v_scalar * bs

        vm = {k: v / v_n for k, v in v_total.items()}

        # Save checkpoint
        if acc.is_main_process:
            is_best = vm["loss"] < best_val
            uw = acc.unwrap_model(model)

            if args.save_every_epoch:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model": uw.state_dict(),
                        "val_loss": vm["loss"],
                        "method": method,
                        "args": vars(args),
                        "train_metrics": {k: float(v) for k, v in tm.items()},
                        "val_metrics": {k: float(v) for k, v in vm.items()},
                    },
                    ckpt_dir / f"epoch_{epoch + 1:03d}.pt",
                )

            if is_best:
                best_val = vm["loss"]
                torch.save({
                    "epoch": epoch + 1,
                    "model": uw.state_dict(),
                    "val_loss": best_val,
                    "method": method,
                    "args": vars(args),
                }, out_dir / "best_model.pt")

            tm_weighted = to_weighted_metrics(tm)
            vm_weighted = to_weighted_metrics(vm)
            log = {"epoch": epoch + 1,
                   **{f"train_{k}": round(v, 6) for k, v in tm_weighted.items()},
                   **{f"val_{k}": round(v, 6) for k, v in vm_weighted.items()}}
            history.append(log)
            
            loss_str = f"[E{epoch + 1:03d}] train={tm['loss']:.4f} val={vm['loss']:.4f}"
            detail_str = f" pair={vm.get('pair', 0.0):.3f} count={vm.get('count', 0.0):.3f}"
            print(f"{loss_str}{detail_str} lr={opt.param_groups[0]['lr']:.2e} {'*BEST' if is_best else ''}")

        if args.stop_after_epoch > 0 and (epoch + 1) >= args.stop_after_epoch:
            if acc.is_main_process:
                print(f"[info] stop_after_epoch reached: {args.stop_after_epoch}")
            break

    # Final save
    if acc.is_main_process:
        uw = acc.unwrap_model(model)
        torch.save({
            "epoch": epoch + 1,
            "model": uw.state_dict(),
            "method": method,
            "args": vars(args),
        }, out_dir / "final_model.pt")
        
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        print(f"\nTraining complete ({method}).")
        print(f"Best validation loss: {best_val:.4f}")
        print(f"Outputs saved to: {out_dir}")

    acc.end_training()


if __name__ == "__main__":
    main()
