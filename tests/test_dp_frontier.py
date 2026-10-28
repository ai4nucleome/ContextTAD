#!/usr/bin/env python3
"""Check the frontier recurrence against independent exhaustive search."""

import importlib.util
import itertools
import logging
import sys
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "1-prepare_data/step2_prepare_labels/scripts/tad_dp_refine.py"
spec = importlib.util.spec_from_file_location("isolated_tad_dp", SOURCE)
dp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dp
spec.loader.exec_module(dp)


class FrontierDPTests(unittest.TestCase):
    def compare_with_enumeration(self, candidates, edges, scorer):
        boundaries = dp.BoundaryMap()
        for bid, values in enumerate(candidates):
            boundaries.boundaries[bid] = dp.UniqueBoundary(
                bid, values[0], [values[0]], values)
        nodes = {
            i: dp.TADNode(i, candidates[left][0], candidates[right][0],
                          candidates[left][0], candidates[right][0], left, right)
            for i, (left, right) in enumerate(edges)
        }
        best_score = dp.NEG_INF
        best_path = None
        for path in itertools.product(*candidates):
            if any(path[left] >= path[right] for left, right in edges):
                continue
            score = sum(scorer(path[left], path[right], nodes[i])
                        for i, (left, right) in enumerate(edges))
            if score > best_score:
                best_score, best_path = score, path
        self.assertIsNotNone(best_path)

        original_scorer = dp.compute_tad_score
        dp.compute_tad_score = lambda left, right, *_args: scorer(
            left, right, _args[-2])
        try:
            dp._optimize_component_dp(
                list(nodes), list(range(len(candidates))), nodes, boundaries,
                None, None, None, None, logging.getLogger("test"))
        finally:
            dp.compute_tad_score = original_scorer
        self.assertEqual(
            tuple(boundaries.boundaries[bid].optimal_coord for bid in range(len(candidates))),
            best_path,
        )

    def test_chain_with_state_merging(self):
        candidates = [[4 * i + delta for delta in (-1, 0, 1)] for i in range(8)]
        edges = [(i, i + 1) for i in range(7)]
        self.compare_with_enumeration(
            candidates, edges,
            lambda left, right, node: -(right - left - 3) ** 2
            - 0.17 * abs(left - node.original_left),
        )

    def test_shared_boundaries_and_invalid_pairs(self):
        candidates = [[3, 4, 5], [4, 5, 6], [5, 6, 7], [7, 8, 9]]
        edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
        self.compare_with_enumeration(
            candidates, edges,
            lambda left, right, node: -(right - left - 2) ** 2
            - 0.01 * node.tad_id * left,
        )

    def test_tie_keeps_first_product_assignment(self):
        candidates = [[1, 2], [3, 4], [5, 6]]
        self.compare_with_enumeration(
            candidates, [(0, 1), (1, 2)], lambda *_args: 0.0)


if __name__ == "__main__":
    unittest.main()
