"""Ensure label construction recomputes DP output instead of reusing old files."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / "1-prepare_data/step2_prepare_labels/scripts/prepare_data.py"
spec = importlib.util.spec_from_file_location("contexttad_prepare_data", SOURCE)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class DPInputTests(unittest.TestCase):
    def test_old_tad_dp_is_replaced_before_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            window = Path(directory)
            (window / "TAD_dp.txt").write_text("99 100\n")

            def write_refined(_input_dir, output_path):
                Path(output_path).write_text("1 6\n2 9\n")

            with patch.object(prepare, "refine_tads", side_effect=write_refined) as refine:
                result = prepare.load_refined_tads(window)
            self.assertEqual(result, [(1, 6), (2, 9)])
            self.assertEqual((window / "TAD_dp.txt").read_text(), "1 6\n2 9\n")
            refine.assert_called_once()

    def test_missing_inputs_fail_instead_of_creating_empty_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                prepare.load_refined_tads(Path(directory) / "missing")

    def test_refinement_failure_does_not_use_old_output(self):
        with tempfile.TemporaryDirectory() as directory:
            window = Path(directory)
            (window / "TAD_dp.txt").write_text("99 100\n")
            with patch.object(prepare, "refine_tads", side_effect=RuntimeError("DP failed")):
                with self.assertRaisesRegex(RuntimeError, "DP failed"):
                    prepare.load_refined_tads(window)
            self.assertEqual((window / "TAD_dp.txt").read_text(), "99 100\n")


if __name__ == "__main__":
    unittest.main()
