"""Regression tests for negative-pair collisions with annotated TADs."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


SOURCE = Path(__file__).resolve().parents[1] / "2-training/core/losses.py"
spec = importlib.util.spec_from_file_location("contexttad_losses", SOURCE)
losses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(losses)


class NegativePairTests(unittest.TestCase):
    def setUp(self):
        self.objective = losses.PairCountLoss()
        self.left = torch.full((1, 12), 0.5, requires_grad=True)
        self.right = torch.tensor([[0.2 + 0.03 * i for i in range(12)]], requires_grad=True)

    def positive_loss(self, pairs):
        values = torch.stack([self.left[0, left] * self.right[0, right] for left, right in pairs])
        return -torch.log(values).mean()

    def test_shared_left_boundary_filters_cyclic_and_jitter_collisions(self):
        pairs = [(1, 6), (1, 7)]
        with patch.object(torch, "randint", return_value=torch.tensor([0, 0])):
            observed = self.objective.pair_loss(self.left, self.right, [torch.tensor(pairs)])
        extra_negative = self.left[0, 1] * self.right[0, 8]
        expected = self.positive_loss(pairs) - 0.5 * torch.log1p(-extra_negative)
        torch.testing.assert_close(observed, expected)

    def test_jitter_into_another_annotated_pair_is_removed(self):
        pairs = [(1, 6), (1, 7)]
        with patch.object(torch, "randint", return_value=torch.tensor([1, -1])):
            observed = self.objective.pair_loss(self.left, self.right, [torch.tensor(pairs)])
        torch.testing.assert_close(observed, self.positive_loss(pairs))

    def test_window_edge_clamp_does_not_relabel_positive(self):
        pairs = [(2, 11)]
        with patch.object(torch, "randint", return_value=torch.tensor([8])):
            observed = self.objective.pair_loss(self.left, self.right, [torch.tensor(pairs)])
        torch.testing.assert_close(observed, self.positive_loss(pairs))

    def test_duplicate_pairs_and_valid_negative_remain_differentiable(self):
        pairs = [(1, 6), (1, 6)]
        with patch.object(torch, "randint", return_value=torch.tensor([1, 1])):
            observed = self.objective.pair_loss(self.left, self.right, [torch.tensor(pairs)])
        extra_negative = self.left[0, 1] * self.right[0, 7]
        expected = self.positive_loss(pairs) - 0.5 * torch.log1p(-extra_negative)
        torch.testing.assert_close(observed, expected)
        observed.backward()
        self.assertTrue(torch.isfinite(self.left.grad).all())
        self.assertTrue(torch.isfinite(self.right.grad).all())


if __name__ == "__main__":
    unittest.main()
