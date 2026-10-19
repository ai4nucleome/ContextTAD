#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from losses import PairCountLoss
from model import ContextTADBackboneModel

PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_DIR = Path(
    os.environ.get("TAD_DATA_DIR", str(PROJECT_ROOT / "0-data" / "1_dp_train_infer_data"))
)
DEFAULT_COVERAGES = ["4000M", "2000M", "1000M", "500M", "250M", "125M", "62_5M"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def normalize_matrix(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)
    lm = np.log1p(arr)
    mx = max(float(lm.max()), 1e-6)
    return np.clip(lm / mx, 0.0, 1.0).astype(np.float32)


class TADWindowDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str,
        use_oe: bool = True,
        coverages: List[str] | None = None,
        sampling_mode: str = "random_single",
        fixed_cov: str = "4000M",
    ):
        self.data_dir = Path(data_dir)
        with (self.data_dir / "window_list.json").open() as f:
            wl = json.load(f)
        self.windows = wl["windows"][split]
        self.split = split
        self.use_oe = bool(use_oe)
        self.covs = coverages or DEFAULT_COVERAGES
        self.sampling_mode = sampling_mode
        self.fixed_cov = fixed_cov
        if sampling_mode == "all_pairs":
            self.index_items = [(w, c) for w in self.windows for c in self.covs]
        else:
            self.index_items = list(self.windows)

    def __len__(self) -> int:
        return len(self.index_items)

    @staticmethod
    def _extract_boundary_counts(tads: np.ndarray, window_size: int = 400):
        left = np.zeros(window_size, dtype=np.float32)
        right = np.zeros(window_size, dtype=np.float32)
        valid = tads[tads[:, 0] >= 0]
        for s, e in valid:
            s, e = int(s), int(e)
            if 0 <= s < window_size:
                left[s] += 1.0
            # *_tads.npy stores closed bin intervals [start, end].
            if 0 <= e < window_size:
                right[e] += 1.0
        return left, right, float(left.sum()), float(right.sum())

    def __getitem__(self, idx: int) -> dict:
        if self.sampling_mode == "all_pairs":
            wn, cov = self.index_items[idx]
        else:
            wn = self.index_items[idx]
            cov = random.choice(self.covs) if self.sampling_mode == "random_single" else self.fixed_cov
        chrom = wn[: wn.rfind("_")]
        mat = np.load(self.data_dir / cov / chrom / f"{wn}.npy")
        arr = mat[1] if self.use_oe else mat[0]
        x = normalize_matrix(arr)
        tads = np.load(self.data_dir / "labels" / chrom / f"{wn}_tads.npy").astype(np.int32)
        left_mask, right_mask, left_count, right_count = self._extract_boundary_counts(tads)
        return {
            "x": torch.from_numpy(x).unsqueeze(0),
            "left_mask": torch.from_numpy((left_mask > 0).astype(np.float32)),
            "right_mask": torch.from_numpy((right_mask > 0).astype(np.float32)),
            "left_count_gt": torch.tensor([left_count], dtype=torch.float32),
            "right_count_gt": torch.tensor([right_count], dtype=torch.float32),
            "tads": torch.from_numpy(tads),
        }


def collate_fn(batch: List[dict]) -> dict:
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "left_mask": torch.stack([b["left_mask"] for b in batch]),
        "right_mask": torch.stack([b["right_mask"] for b in batch]),
        "left_count_gt": torch.stack([b["left_count_gt"] for b in batch]),
        "right_count_gt": torch.stack([b["right_count_gt"] for b in batch]),
        "tads_list": [b["tads"] for b in batch],
    }


def warmup_cosine_lr(step: int, total: int, warmup: int, init_lr: float, base_lr: float, min_lr: float) -> float:
    """Linear warmup followed by cosine decay, matching backup/2-mutbert/utils.py."""
    if step < warmup:
        progress = float(step) / max(float(warmup), 1.0)
        return init_lr + (base_lr - init_lr) * progress
    progress = float(step - warmup) / max(float(total - warmup), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def warmup_cosine_factor(step: int, total: int, warmup: int, init_lr: float, base_lr: float, min_lr: float) -> float:
    return warmup_cosine_lr(step, total, warmup, init_lr, base_lr, min_lr) / max(base_lr, 1e-12)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ContextTAD residual U-Net model with Hugging Face Accelerate")
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--init_lr", type=float, default=1e-8, help="Initial LR before warmup")
    p.add_argument("--min_lr", type=float, default=1e-8, help="Minimum LR after cosine decay")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_epochs", type=float, default=1.0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--diag_radius", type=int, default=50, help="Near-diagonal radius in bins used by the 2D-to-1D projection")
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--use_obs", action="store_true", help="Use observed matrix instead of O/E")
    p.add_argument("--train_sampling", default="random_single", choices=["random_single", "all_pairs", "fixed_single"])
    p.add_argument("--val_sampling", default="random_single", choices=["random_single", "all_pairs", "fixed_single"])
    p.add_argument("--val_fixed_cov", default="4000M")
    p.add_argument("--pair_weight", type=float, default=1.0)
    p.add_argument("--count_weight", type=float, default=32.0)
    p.add_argument("--boundary_weight", type=float, default=0.0)
    p.add_argument("--pair_hard_window", type=int, default=8)
    p.add_argument("--max_grad_norm", type=float, default=0.4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_train_steps", type=int, default=0)
    p.add_argument("--max_val_steps", type=int, default=0)
    p.add_argument("--save_every_epoch", action="store_true")
    p.add_argument("--tb_log_every", type=int, default=50, help="TensorBoard train-step logging interval; 0 disables step-level logging")
    return p.parse_args()


def metric_vector(losses: dict, batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            float(losses["loss"].detach()) * batch_size,
            float(losses["pair"].detach()) * batch_size,
            float(losses["count"].detach()) * batch_size,
            float(losses["boundary"].detach()) * batch_size,
            float(batch_size),
        ],
        device=device,
        dtype=torch.float32,
    )


def main() -> None:
    args = parse_args()
    args.backbone = "unet_decoder"
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum_steps,
        log_with="tensorboard",
        project_dir=str(Path(args.output_dir)),
    )
    set_seed(args.seed + accelerator.process_index)

    data_dir = Path(args.data_dir)
    if not (data_dir / "window_list.json").exists():
        raise FileNotFoundError(f"Missing data directory or window_list.json: {data_dir}")
    out_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.save_every_epoch:
            (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        with (out_dir / "args.json").open("w") as f:
            json.dump(vars(args), f, indent=2)
    accelerator.wait_for_everyone()

    tracker_name = f"tensorboard_{out_dir.name}"
    experiment_config = vars(args).copy()
    accelerator.init_trackers(tracker_name, experiment_config)

    train_ds = TADWindowDataset(data_dir, "train", use_oe=not args.use_obs, sampling_mode=args.train_sampling)
    val_ds = TADWindowDataset(data_dir, "val", use_oe=not args.use_obs, sampling_mode=args.val_sampling, fixed_cov=args.val_fixed_cov)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = ContextTADBackboneModel(hidden=args.hidden, diag_radius=args.diag_radius)
    criterion = PairCountLoss(
        pair_weight=args.pair_weight,
        count_weight=args.count_weight,
        boundary_weight=args.boundary_weight,
        pair_hard_window=args.pair_hard_window,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    steps_per_epoch = args.max_train_steps if args.max_train_steps > 0 else math.ceil(len(train_dl) / max(accelerator.num_processes, 1))
    total_steps = max(math.ceil(steps_per_epoch * args.epochs / max(args.grad_accum_steps, 1)), 1)
    warmup_steps = max(int(math.ceil(steps_per_epoch * args.warmup_epochs / max(args.grad_accum_steps, 1))), 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: warmup_cosine_factor(s, total_steps, warmup_steps, args.init_lr, args.lr, args.min_lr))

    model, optimizer, train_dl, val_dl, scheduler = accelerator.prepare(model, optimizer, train_dl, val_dl, scheduler)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print("Backbone: residual U-Net decoder")
    accelerator.print(f"Diagonal projection radius: {args.diag_radius} bins (dw={2 * args.diag_radius + 1})")
    accelerator.print(f"Trainable parameters: {n_params/1e6:.2f}M")
    accelerator.print(f"Accelerate: processes={accelerator.num_processes} mixed_precision={accelerator.mixed_precision} grad_accum={args.grad_accum_steps}")
    accelerator.print(f"Data: {data_dir}")
    accelerator.print(f"Train windows: {len(train_ds)} | Val windows: {len(val_ds)}")
    accelerator.print(
        f"Loss weights: pair={args.pair_weight}, count={args.count_weight}, "
        f"boundary={args.boundary_weight}"
    )
    accelerator.print(f"Optimizer: AdamW init_lr={args.init_lr} base_lr={args.lr} min_lr={args.min_lr} wd={args.weight_decay} warmup_steps={warmup_steps} total_steps={total_steps}")

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum = torch.zeros(5, device=accelerator.device)
        pbar = tqdm(train_dl, desc=f"E{epoch}", leave=False, disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(pbar, start=1):
            if args.max_train_steps and step > args.max_train_steps:
                break
            with accelerator.accumulate(model):
                outputs = model(batch["x"])
                targets = {
                    "left_mask": batch["left_mask"],
                    "right_mask": batch["right_mask"],
                    "left_count_gt": batch["left_count_gt"],
                    "right_count_gt": batch["right_count_gt"],
                }
                tads_list = [t.to(accelerator.device, non_blocking=True) for t in batch["tads_list"]]
                losses = criterion(outputs, targets, tads_list)
                accelerator.backward(losses["loss"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if not getattr(accelerator, "optimizer_step_was_skipped", False):
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            bs = int(batch["x"].shape[0])
            gathered = accelerator.gather_for_metrics(metric_vector(losses, bs, accelerator.device).unsqueeze(0))
            train_sum += gathered.sum(dim=0)
            if accelerator.is_local_main_process:
                pbar.set_postfix(loss=f"{float(losses['loss'].detach()):.4f}")
            if accelerator.is_main_process and args.tb_log_every > 0 and accelerator.sync_gradients:
                opt_step = int(scheduler.scheduler.last_epoch if hasattr(scheduler, "scheduler") else scheduler.last_epoch)
                if opt_step % args.tb_log_every == 0:
                    accelerator.log(
                        {
                            "train_step/loss": float(losses["loss"].detach()),
                            "train_step/pair": float(losses["pair"].detach()),
                            "train_step/count": float(losses["count"].detach()),
                            "train_step/boundary": float(losses["boundary"].detach()),
                            "train_step/lr": optimizer.param_groups[0]["lr"],
                        },
                        step=opt_step,
                    )
        train_metrics = {
            "loss": float(train_sum[0].item() / max(train_sum[4].item(), 1.0)),
            "pair": float(train_sum[1].item() / max(train_sum[4].item(), 1.0)),
            "count": float(train_sum[2].item() / max(train_sum[4].item(), 1.0)),
            "boundary": float(train_sum[3].item() / max(train_sum[4].item(), 1.0)),
        }

        model.eval()
        val_sum = torch.zeros(5, device=accelerator.device)
        with torch.no_grad():
            for step, batch in enumerate(tqdm(val_dl, desc="val", leave=False, disable=not accelerator.is_local_main_process), start=1):
                if args.max_val_steps and step > args.max_val_steps:
                    break
                outputs = model(batch["x"])
                targets = {
                    "left_mask": batch["left_mask"],
                    "right_mask": batch["right_mask"],
                    "left_count_gt": batch["left_count_gt"],
                    "right_count_gt": batch["right_count_gt"],
                }
                tads_list = [t.to(accelerator.device, non_blocking=True) for t in batch["tads_list"]]
                losses = criterion(outputs, targets, tads_list)
                bs = int(batch["x"].shape[0])
                gathered = accelerator.gather_for_metrics(metric_vector(losses, bs, accelerator.device).unsqueeze(0))
                val_sum += gathered.sum(dim=0)
        val_metrics = {
            "loss": float(val_sum[0].item() / max(val_sum[4].item(), 1.0)),
            "pair": float(val_sum[1].item() / max(val_sum[4].item(), 1.0)),
            "count": float(val_sum[2].item() / max(val_sum[4].item(), 1.0)),
            "boundary": float(val_sum[3].item() / max(val_sum[4].item(), 1.0)),
        }

        is_best = val_metrics["loss"] < best_val
        if is_best:
            best_val = val_metrics["loss"]
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            state = {
                "epoch": epoch,
                "model": accelerator.get_state_dict(model),
                "args": vars(args),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
            if is_best:
                accelerator.save(state, out_dir / "best_model.pt")
            if args.save_every_epoch:
                accelerator.save(state, out_dir / "checkpoints" / f"epoch_{epoch:03d}.pt")
            history.append({
                "epoch": epoch,
                **{f"train_{k}": round(v, 6) for k, v in train_metrics.items()},
                **{f"val_{k}": round(v, 6) for k, v in val_metrics.items()},
                "lr": optimizer.param_groups[0]["lr"],
            })
            accelerator.log(
                {
                    "epoch/train_loss": train_metrics["loss"],
                    "epoch/train_pair": train_metrics["pair"],
                    "epoch/train_count": train_metrics["count"],
                    "epoch/train_boundary": train_metrics["boundary"],
                    "epoch/val_loss": val_metrics["loss"],
                    "epoch/val_pair": val_metrics["pair"],
                    "epoch/val_count": val_metrics["count"],
                    "epoch/val_boundary": val_metrics["boundary"],
                    "epoch/lr": optimizer.param_groups[0]["lr"],
                    "epoch/best_val_loss": best_val,
                },
                step=epoch,
            )
            with (out_dir / "history.json").open("w") as f:
                json.dump(history, f, indent=2)
            print(
                f"[E{epoch:03d}] train={train_metrics['loss']:.4f} val={val_metrics['loss']:.4f} "
                f"pair={val_metrics['pair']:.4f} count={val_metrics['count']:.4f} "
                f"boundary={val_metrics['boundary']:.4f} lr={optimizer.param_groups[0]['lr']:.2e} "
                f"{'*BEST' if is_best else ''}"
            )
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        accelerator.save({"epoch": args.epochs, "model": accelerator.get_state_dict(model), "args": vars(args)}, out_dir / "final_model.pt")
        print(f"Training complete. best_val={best_val:.4f} out_dir={out_dir}")
    accelerator.end_training()


if __name__ == "__main__":
    main()
