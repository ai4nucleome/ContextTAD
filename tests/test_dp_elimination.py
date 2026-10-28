#!/usr/bin/env python3
"""Compare variable-elimination DP with independent exhaustive search."""

import importlib.util
import itertools
import logging
import sys
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "1-prepare_data/step2_prepare_labels/scripts/tad_dp_refine.py"
spec = importlib.util.spec_from_file_location("isolated_tad_all_dp", SOURCE)
dp = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = dp
spec.loader.exec_module(dp)


class VariableEliminationTests(unittest.TestCase):
    def check_case(self, candidates, edges, scorer):
        boundaries = dp.BoundaryMap()
        for bid, values in enumerate(candidates):
            boundaries.boundaries[bid] = dp.UniqueBoundary(
                bid, values[0], [values[0]], values)
        nodes = {
            i: dp.TADNode(i, candidates[left][0], candidates[right][0],
                          candidates[left][0], candidates[right][0], left, right)
            for i, (left, right) in enumerate(edges)
        }
        optimum = float("-inf")
        best_paths = []
        for path in itertools.product(*candidates):
            if any(path[left] >= path[right] for left, right in edges):
                continue
            value = sum(scorer(path[left], path[right], nodes[i])
                        for i, (left, right) in enumerate(edges))
            if value > optimum:
                optimum, best_paths = value, [path]
            elif value == optimum:
                best_paths.append(path)
        self.assertTrue(best_paths)

        original_scorer = dp.compute_tad_score
        dp.compute_tad_score = lambda left, right, *_args: scorer(left, right, _args[-2])
        try:
            dp._optimize_component_elimination(
                list(nodes), list(range(len(candidates))), nodes, boundaries,
                None, None, None, None, logging.getLogger("test"))
        finally:
            dp.compute_tad_score = original_scorer
        observed = tuple(boundaries.boundaries[i].optimal_coord
                         for i in range(len(candidates)))
        self.assertIn(observed, best_paths)

    def test_eight_boundary_chain(self):
        candidates = [[4 * i + delta for delta in (-1, 0, 1)] for i in range(8)]
        self.check_case(candidates, [(i, i + 1) for i in range(7)],
                        lambda left, right, node: -(right - left - 3) ** 2
                        - 0.17 * abs(left - node.original_left))

    def test_nested_shared_boundaries(self):
        candidates = [[3, 4, 5], [4, 5, 6], [5, 6, 7], [7, 8, 9]]
        self.check_case(candidates, [(0, 1), (0, 2), (1, 3), (2, 3)],
                        lambda left, right, node: -(right - left - 2) ** 2
                        - 0.01 * node.tad_id * left)


if __name__ == "__main__":
    unittest.main()
