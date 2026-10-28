#!/usr/bin/env python3
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairCountLoss(nn.Module):
    """ContextTAD objectives and their published ablation switches."""

    def __init__(
        self,
        pair_weight: float = 1.0,
        count_weight: float = 32.0,
        boundary_weight: float = 0.0,
        min_len_bins: int = 5,
        pair_hard_window: int = 8,
    ):
        super().__init__()
        self.pair_weight = float(pair_weight)
        self.count_weight = float(count_weight)
        self.boundary_weight = float(boundary_weight)
        self.min_len = int(min_len_bins)
        self.pair_hard_window = int(pair_hard_window)

    def pair_loss(
        self,
        left_pred: torch.Tensor,
        right_pred: torch.Tensor,
        tads_list: List[torch.Tensor],
    ) -> torch.Tensor:
        left_pred = left_pred.float()
        right_pred = right_pred.float()
        batch_size, n_bins = left_pred.shape
        total = left_pred.new_tensor(0.0)
        valid_samples = 0

        for batch_idx in range(batch_size):
            tads = tads_list[batch_idx]
            if tads is None or len(tads) == 0:
                continue
            tads = tads.to(left_pred.device)
            valid_tads = tads[tads[:, 0] >= 0]
            if len(valid_tads) == 0:
                continue

            left_idx = valid_tads[:, 0].long()
            right_idx = valid_tads[:, 1].long()
            valid = (left_idx < n_bins) & (right_idx < n_bins) & (left_idx < right_idx)
            left_idx = left_idx[valid]
            right_idx = right_idx[valid]
            if len(left_idx) == 0:
                continue

            positive = (
                left_pred[batch_idx, left_idx] * right_pred[batch_idx, right_idx]
            ).clamp(1e-6, 1.0 - 1e-6)
            positive_loss = -torch.log(positive).mean()
            positive_pair_ids = left_idx * n_bins + right_idx

            negative_parts = []
            if len(left_idx) > 1:
                shifted_right = torch.roll(right_idx, shifts=1)
                valid_mismatch = (
                    (shifted_right > left_idx)
                    & (shifted_right - left_idx >= self.min_len)
                    & ~torch.isin(left_idx * n_bins + shifted_right, positive_pair_ids)
                )
                if valid_mismatch.any():
                    negative_parts.append(
                        (
                            left_pred[batch_idx, left_idx[valid_mismatch]]
                            * right_pred[batch_idx, shifted_right[valid_mismatch]]
                        ).clamp(1e-6, 1.0 - 1e-6)
                    )

            jitter = torch.randint(
                -self.pair_hard_window,
                self.pair_hard_window + 1,
                right_idx.shape,
                device=left_pred.device,
            )
            jitter = torch.where(jitter == 0, torch.ones_like(jitter), jitter)
            jitter_right = (right_idx + jitter).clamp(0, n_bins - 1)
            valid_jitter = (
                (jitter_right > (left_idx + self.min_len - 1))
                & ~torch.isin(left_idx * n_bins + jitter_right, positive_pair_ids)
            )
            if valid_jitter.any():
                negative_parts.append(
                    (
                        left_pred[batch_idx, left_idx[valid_jitter]]
                        * right_pred[batch_idx, jitter_right[valid_jitter]]
                    ).clamp(1e-6, 1.0 - 1e-6)
                )

            if negative_parts:
                negative = torch.cat(negative_parts)
                negative_loss = -torch.log1p(-negative).mean()
            else:
                negative_loss = left_pred.new_tensor(0.0)

            total = total + positive_loss + 0.5 * negative_loss
            valid_samples += 1

        return total / max(valid_samples, 1)

    @staticmethod
    def count_loss(
        left_pred: torch.Tensor,
        right_pred: torch.Tensor,
        left_count_gt: torch.Tensor,
        right_count_gt: torch.Tensor,
    ) -> torch.Tensor:
        pred_left = left_pred.float().sum(dim=1)
        pred_right = right_pred.float().sum(dim=1)
        gt_left = left_count_gt.float().squeeze(-1)
        gt_right = right_count_gt.float().squeeze(-1)
        n_bins = left_pred.shape[-1]
        return (F.l1_loss(pred_left, gt_left) + F.l1_loss(pred_right, gt_right)) / n_bins

    @staticmethod
    def boundary_loss(
        left_pred: torch.Tensor,
        right_pred: torch.Tensor,
        left_mask: torch.Tensor,
        right_mask: torch.Tensor,
    ) -> torch.Tensor:
        left_pred = left_pred.float().clamp(1e-6, 1.0 - 1e-6)
        right_pred = right_pred.float().clamp(1e-6, 1.0 - 1e-6)
        return F.binary_cross_entropy(left_pred, left_mask.float()) + F.binary_cross_entropy(
            right_pred, right_mask.float()
        )

    def forward(self, outputs: dict, targets: dict, tads_list: List[torch.Tensor]) -> dict:
        left = outputs["left_boundary"]
        right = outputs["right_boundary"]
        zero = (left.sum() + right.sum()) * 0.0

        pair = self.pair_loss(left, right, tads_list) if self.pair_weight > 0 else zero
        count = (
            self.count_loss(left, right, targets["left_count_gt"], targets["right_count_gt"])
            if self.count_weight > 0
            else zero
        )
        boundary = (
            self.boundary_loss(left, right, targets["left_mask"], targets["right_mask"])
            if self.boundary_weight > 0
            else zero
        )
        loss = (
            self.pair_weight * pair
            + self.count_weight * count
            + self.boundary_weight * boundary
        )
        return {"loss": loss, "pair": pair, "count": count, "boundary": boundary}
