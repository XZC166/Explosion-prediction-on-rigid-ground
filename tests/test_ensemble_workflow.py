import unittest
import uuid
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from evaluate_metrics import evaluate_directory, evaluate_runs, print_report
from main_case_loop_ensemble import (
    ensemble_member_predictions,
    load_split_manifest,
    parse_dropout_values,
    write_prediction_file,
)


TEST_WORKSPACE = Path.cwd() / "_test_workspace"


def reset_test_dir(name):
    root = TEST_WORKSPACE / f"{name}_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    return root


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.array(rows, dtype=float), fmt="%.4f")


class EvaluateMetricsTests(unittest.TestCase):
    def test_evaluate_runs_accepts_configurable_prediction_base_and_five_column_files(self):
        root = reset_test_dir("evaluate_five_column")
        true_dir = root / "truth"
        pred_base = root / "predictions"

        write_rows(true_dir / "value1", [[0, 0, 0, 101425], [1, 0, 0, 101525]])
        write_rows(pred_base / "run_1" / "test" / "value1", [[0, 0, 0, 101475, 10], [1, 0, 0, 101575, 20]])

        single = evaluate_directory(str(true_dir), str(pred_base / "run_1" / "test"))
        runs = evaluate_runs(str(true_dir), str(pred_base), num_runs=1)

        self.assertEqual(single["samples"], 2)
        expected_mape = np.mean([50.0 / (100.0 + 1e-5), 50.0 / (200.0 + 1e-5)]) * 100
        self.assertAlmostEqual(single["mape_over"], expected_mape, places=6)
        self.assertEqual(len(runs["rows"]), 1)
        self.assertEqual(runs["rows"][0]["run_id"], 1)
        self.assertEqual(runs["rows"][0]["split"], "Test")
        self.assertAlmostEqual(runs["test_avg_mape"], expected_mape, places=6)

    def test_print_report_uses_requested_run_count(self):
        summary = {
            "num_runs": 1,
            "rows": [],
            "train_avg_mape": 0.0,
            "test_avg_mape": 0.0,
            "test_avg_r2": 0.0,
        }
        output = StringIO()

        with redirect_stdout(output):
            print_report(summary)

        self.assertIn("1次交叉验证平均", output.getvalue())


class EnsembleWorkflowTests(unittest.TestCase):
    def test_parse_dropout_values(self):
        self.assertEqual(parse_dropout_values("0.0,0.1, 0.5"), [0.0, 0.1, 0.5])

    def test_load_split_manifest_uses_existing_prediction_split_files(self):
        root = reset_test_dir("split_manifest")
        (root / "predictions" / "run_1" / "train").mkdir(parents=True)
        (root / "predictions" / "run_1" / "test").mkdir(parents=True)
        (root / "predictions" / "run_1" / "train" / "value1").touch()
        (root / "predictions" / "run_1" / "test" / "value2").touch()

        manifest = load_split_manifest(root / "predictions", 1)

        self.assertEqual(manifest["train"], ["value1"])
        self.assertEqual(manifest["test"], ["value2"])

    def test_ensemble_member_predictions_returns_mean_and_member_variance(self):
        mean, var = ensemble_member_predictions(
            [
                np.array([[101.0], [201.0]]),
                np.array([[103.0], [205.0]]),
                np.array([[105.0], [209.0]]),
            ]
        )

        np.testing.assert_allclose(mean, np.array([103.0, 205.0]))
        np.testing.assert_allclose(var, np.array([8.0 / 3.0, 32.0 / 3.0]))

    def test_write_prediction_file_writes_mean_and_variance_columns(self):
        root = reset_test_dir("write_prediction_file")
        output_path = root / "value1"
        raw_points = np.array([[0.0, 1.0, 2.0, 999.0], [3.0, 4.0, 5.0, 999.0]])
        write_prediction_file(output_path, raw_points, np.array([10.0, 20.0]), np.array([1.5, 2.5]))

        data = np.loadtxt(output_path)

        self.assertEqual(data.shape, (2, 5))
        np.testing.assert_allclose(data[:, 0:3], raw_points[:, 0:3])
        np.testing.assert_allclose(data[:, 3], [10.0, 20.0])
        np.testing.assert_allclose(data[:, 4], [1.5, 2.5])


if __name__ == "__main__":
    unittest.main()
