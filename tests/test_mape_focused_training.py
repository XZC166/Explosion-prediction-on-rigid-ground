import argparse
import unittest
import uuid
from pathlib import Path

import numpy as np

from evaluate_metrics import evaluate_directory
from main_case_loop_ensemble import load_split_manifest
from mape_focused_training import (
    DEFAULT_CONFIGS,
    assert_output_split_matches_manifest,
    compute_z_sample_weights,
    log_target_to_overpressure,
    overpressure_to_log_target,
    parse_config_names,
    write_single_model_prediction_file,
)


TEST_WORKSPACE = Path.cwd() / "_test_workspace"


def reset_test_dir(name):
    root = TEST_WORKSPACE / f"{name}_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    return root


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.array(rows, dtype=float), fmt="%.4f")


class MapeFocusedTrainingTests(unittest.TestCase):
    def test_log_target_roundtrip_preserves_overpressure_values(self):
        overpressure = np.array([0.0, 1.0, 1000.0, 250000.0])

        restored = log_target_to_overpressure(overpressure_to_log_target(overpressure))

        np.testing.assert_allclose(restored, overpressure, rtol=1e-10, atol=1e-8)

    def test_z_weights_boost_high_z_samples_and_apply_cap(self):
        z_values = np.array([100.0, 3000.0, 6000.0, 12000.0])

        weights = compute_z_sample_weights(z_values, threshold=3500.0, max_weight=3.0)

        self.assertAlmostEqual(weights[0], 1.0)
        self.assertEqual(float(np.max(weights)), 3.0)
        self.assertGreater(weights[2], weights[1])
        self.assertGreater(weights[3], weights[2])

    def test_parse_config_names_defaults_and_rejects_unknown(self):
        self.assertEqual(parse_config_names(None), DEFAULT_CONFIGS)
        self.assertEqual(parse_config_names("log_l1, log_l1_z_weighted"), ["log_l1", "log_l1_z_weighted"])

        with self.assertRaises(ValueError):
            parse_config_names("log_l1,unknown_config")

    def test_single_model_prediction_file_has_five_columns_and_evaluates(self):
        root = reset_test_dir("single_prediction")
        true_dir = root / "truth"
        pred_dir = root / "predictions"
        raw_points = np.array([[0.0, 0.0, 0.0, 101425.0], [1.0, 0.0, 0.0, 101525.0]])

        write_rows(true_dir / "value1", raw_points)
        write_single_model_prediction_file(pred_dir / "value1", raw_points, np.array([101475.0, 101575.0]))

        pred_data = np.loadtxt(pred_dir / "value1")
        result = evaluate_directory(str(true_dir), str(pred_dir))

        self.assertEqual(pred_data.shape, (2, 5))
        np.testing.assert_allclose(pred_data[:, 4], [0.0, 0.0])
        self.assertEqual(result["samples"], 2)

    def test_output_split_must_match_reused_baseline_manifest(self):
        root = reset_test_dir("manifest_match")
        baseline = root / "baseline"
        output = root / "output"
        (baseline / "run_1" / "train").mkdir(parents=True)
        (baseline / "run_1" / "test").mkdir(parents=True)
        (baseline / "run_1" / "train" / "value1").touch()
        (baseline / "run_1" / "test" / "value2").touch()
        (output / "run_1" / "train").mkdir(parents=True)
        (output / "run_1" / "test").mkdir(parents=True)
        (output / "run_1" / "train" / "value1").touch()
        (output / "run_1" / "test" / "value2").touch()

        manifest = load_split_manifest(baseline, 1)
        assert_output_split_matches_manifest(output / "run_1", manifest)

        (output / "run_1" / "test" / "extra_value").touch()
        with self.assertRaises(RuntimeError):
            assert_output_split_matches_manifest(output / "run_1", manifest)


if __name__ == "__main__":
    unittest.main()
