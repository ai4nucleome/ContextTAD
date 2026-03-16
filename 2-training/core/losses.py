from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TADStructureLoss(nn.Module):
    """Minimal active loss: pair consistency + boundary-count regression."""

    def __init__(
        self,
        pair_weight: float = 1.0,
        count_weight: float = 0.3,
        min_len_bins: int = 5,
        max_len_bins: int = 200,
        pair_hard_window: int = 8,
        use_pair_hard_negatives: bool = True,
    ):
        super().__init__()
        self.w_pair = float(pair_weight)
        self.w_count = float(count_weight)
        self.min_len = int(min_len_bins)
        self.max_len = int(max_len_bins)
        self.pair_hard_window = int(pair_hard_window)
        self.use_pair_hard_negatives = bool(use_pair_hard_negatives)

    def _pair_loss(
        self,
        left_pred: torch.Tensor,
        right_pred: torch.Tensor,
        tads_list: list,
    ) -> torch.Tensor:
        bsz, n_bins = left_pred.shape
        device = left_pred.device
        total_loss = left_pred.new_tensor(0.0)
        valid_samples = 0

        for b in range(bsz):
            tads = tads_list[b]
            if tads is None or len(tads) == 0:
                continue
            valid_tads = tads[tads[:, 0] >= 0]
            if len(valid_tads) == 0:
                continue

            left_idx = valid_tads[:, 0].long()
            right_idx = valid_tads[:, 1].long()
            valid_pairs = (left_idx < n_bins) & (right_idx < n_bins) & (left_idx < right_idx)
            left_idx = left_idx[valid_pairs]
            right_idx = right_idx[valid_pairs]
            if len(left_idx) == 0:
                continue

            left_scores = left_pred[b, left_idx]
            right_scores = right_pred[b, right_idx]
            pos_scores = (left_scores * right_scores).clamp(1e-6, 1 - 1e-6)
            pos_loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))

            neg_parts = []
            if len(left_idx) > 1:
                shifted_right = torch.roll(right_idx, shifts=1)
                neg_parts.append((left_pred[b, left_idx] * right_pred[b, shifted_right]).clamp(1e-6, 1 - 1e-6))

            if self.use_pair_hard_negatives and self.pair_hard_window > 0:
                jitter = torch.randint(
                    low=-self.pair_hard_window,
                    high=self.pair_hard_window + 1,
                    size=right_idx.shape,
                    device=device,
                )
                jitter = torch.where(jitter == 0, torch.ones_like(jitter), jitter)
                jitter_right = (right_idx + jitter).clamp(min=0, max=n_bins - 1)
                valid_jitter = jitter_right > (left_idx + self.min_len - 1)
                if valid_jitter.any():
                    neg_parts.append(
                        (left_pred[b, left_idx[valid_jitter]] * right_pred[b, jitter_right[valid_jitter]]).clamp(
                            1e-6, 1 - 1e-6
                        )
                    )

            if neg_parts:
                neg_scores = torch.cat(neg_parts)
                neg_loss = F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
            else:
                neg_loss = left_pred.new_tensor(0.0)

            total_loss = total_loss + pos_loss + 0.5 * neg_loss
            valid_samples += 1

        return total_loss / max(valid_samples, 1)

    def _count_loss(
        self,
        left_pred: torch.Tensor,
        right_pred: torch.Tensor,
        left_count_gt: torch.Tensor,
        right_count_gt: torch.Tensor,
    ) -> torch.Tensor:
        pred_left_count = left_pred.sum(dim=1)
        pred_right_count = right_pred.sum(dim=1)
        gt_left_count = left_count_gt.float().squeeze(-1)
        gt_right_count = right_count_gt.float().squeeze(-1)
        return F.l1_loss(pred_left_count, gt_left_count) + F.l1_loss(pred_right_count, gt_right_count)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        tads_list: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        left = outputs["left_boundary"]
        right = outputs["right_boundary"]
        device = left.device

        if self.w_pair > 0 and tads_list is not None:
            pair_loss = self._pair_loss(left, right, tads_list)
        else:
            pair_loss = torch.tensor(0.0, device=device)

        if self.w_count > 0 and "left_count_gt" in targets and "right_count_gt" in targets:
            count_loss = self._count_loss(left, right, targets["left_count_gt"], targets["right_count_gt"])
        else:
            count_loss = torch.tensor(0.0, device=device)

        total = self.w_pair * pair_loss + self.w_count * count_loss
        return {
            "pair": pair_loss,
            "count": count_loss,
            "loss": total,
        }
