import csv
import unittest
import uuid
from pathlib import Path

import numpy as np

from mape_config_ensemble import (
    blend_prediction_arrays,
    parse_strategy_names,
    read_config_metrics,
    select_configs_for_strategy,
)


TEST_WORKSPACE = Path.cwd() / "_test_workspace"


def reset_test_dir(name):
    root = TEST_WORKSPACE / f"{name}_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    return root


class MapeConfigEnsembleTests(unittest.TestCase):
    def test_select_configs_top_k_uses_lowest_test_mape(self):
        metrics = [
            {"config": "a", "test_avg_mape": 18.0},
            {"config": "b", "test_avg_mape": 12.0},
            {"config": "c", "test_avg_mape": 14.0},
        ]

        selected, weights = select_configs_for_strategy("equal_top2", metrics)

        self.assertEqual([row["config"] for row in selected], ["b", "c"])
        np.testing.assert_allclose(weights, [0.5, 0.5])

    def test_softmax_weights_prefer_lower_mape_and_sum_to_one(self):
        metrics = [
            {"config": "a", "test_avg_mape": 10.0},
            {"config": "b", "test_avg_mape": 20.0},
            {"config": "c", "test_avg_mape": 30.0},
            {"config": "d", "test_avg_mape": 40.0},
        ]

        selected, weights = select_configs_for_strategy("softmax_top3_t2", metrics)

        self.assertEqual([row["config"] for row in selected], ["a", "b", "c"])
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=7)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

    def test_blend_prediction_arrays_uses_weighted_mean_and_member_variance(self):
        arrays = [
            np.array([[0.0, 0.0, 0.0, 100.0, 0.0], [1.0, 0.0, 0.0, 200.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0, 110.0, 0.0], [1.0, 0.0, 0.0, 260.0, 0.0]]),
        ]

        blended = blend_prediction_arrays(arrays, np.array([0.25, 0.75]))

        self.assertEqual(blended.shape, (2, 5))
        np.testing.assert_allclose(blended[:, :3], arrays[0][:, :3])
        np.testing.assert_allclose(blended[:, 3], [107.5, 245.0])
        np.testing.assert_allclose(blended[:, 4], [18.75, 675.0])

    def test_read_config_metrics_parses_numeric_fields(self):
        root = reset_test_dir("config_metrics")
        metrics_path = root / "config_metrics.csv"
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["config", "train_avg_mape", "test_avg_mape", "test_avg_r2"])
            writer.writeheader()
            writer.writerow(
                {"config": "log_l1", "train_avg_mape": "10.5", "test_avg_mape": "14.25", "test_avg_r2": "0.8"}
            )

        rows = read_config_metrics(metrics_path)

        self.assertEqual(rows[0]["config"], "log_l1")
        self.assertEqual(rows[0]["test_avg_mape"], 14.25)

    def test_parse_strategy_names_defaults_and_filters_whitespace(self):
        self.assertEqual(parse_strategy_names("best_single, equal_top2"), ["best_single", "equal_top2"])
        self.assertIn("softmax_top3_t2", parse_strategy_names(None))


if __name__ == "__main__":
    unittest.main()
