import csv
import unittest
import uuid
from pathlib import Path

import numpy as np

from ensemble_weighted_inference import (
    DEFAULT_BEST_STRATEGY,
    DEFAULT_STRATEGIES,
    compute_weighted_mean_and_variance,
    parse_strategy_names,
    resolve_requested_strategies,
    read_member_metrics,
    select_members_for_strategy,
)


TEST_WORKSPACE = Path.cwd() / "_test_workspace"


def reset_test_dir(name):
    root = TEST_WORKSPACE / f"{name}_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    return root


class WeightedInferenceTests(unittest.TestCase):
    def test_select_members_for_top_k_uses_lowest_validation_mape(self):
        members = [(0.0, 15.0), (0.1, 18.0), (0.2, 12.0), (0.3, 14.0)]

        selected, weights = select_members_for_strategy("top2", members)

        self.assertEqual([p for p, _ in selected], [0.2, 0.3])
        np.testing.assert_allclose(weights, [0.5, 0.5])

    def test_softmax_top3_weights_prefer_lower_mape_and_sum_to_one(self):
        members = [(0.0, 10.0), (0.1, 20.0), (0.2, 30.0), (0.3, 40.0)]

        selected, weights = select_members_for_strategy("softmax_top3_t5", members)

        self.assertEqual([p for p, _ in selected], [0.0, 0.1, 0.2])
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=7)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

    def test_inverse_mape_top3_weights_are_normalized(self):
        members = [(0.0, 10.0), (0.1, 20.0), (0.2, 40.0), (0.3, 100.0)]

        selected, weights = select_members_for_strategy("inv_mape_top3", members)

        self.assertEqual([p for p, _ in selected], [0.0, 0.1, 0.2])
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=7)
        np.testing.assert_allclose(weights, np.array([0.1, 0.05, 0.025]) / 0.175)

    def test_compute_weighted_mean_and_variance_uses_selected_member_weights(self):
        predictions = [
            np.array([10.0, 20.0]),
            np.array([14.0, 24.0]),
            np.array([18.0, 28.0]),
        ]
        weights = np.array([0.2, 0.3, 0.5])

        mean, variance = compute_weighted_mean_and_variance(predictions, weights)

        np.testing.assert_allclose(mean, [15.2, 25.2])
        expected_variance = np.average(
            (np.stack(predictions, axis=0) - np.array([[15.2, 25.2]])) ** 2,
            axis=0,
            weights=weights,
        )
        np.testing.assert_allclose(variance, expected_variance)

    def test_parse_strategy_names_defaults_and_filters_whitespace(self):
        self.assertEqual(parse_strategy_names(None), DEFAULT_STRATEGIES)
        self.assertEqual(parse_strategy_names("top2, softmax_top3_t2"), ["top2", "softmax_top3_t2"])

    def test_best_only_does_not_override_explicit_strategies(self):
        self.assertEqual(resolve_requested_strategies("top2", best_only=True), ["top2"])
        self.assertEqual(resolve_requested_strategies(None, best_only=True), [DEFAULT_BEST_STRATEGY])

    def test_read_member_metrics_groups_by_run(self):
        root = reset_test_dir("member_metrics")
        csv_path = root / "member_metrics.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["run_id", "dropout_p", "best_val_mape_percent"])
            writer.writeheader()
            writer.writerow({"run_id": 1, "dropout_p": 0.0, "best_val_mape_percent": 15.0})
            writer.writerow({"run_id": 1, "dropout_p": 0.2, "best_val_mape_percent": 12.0})
            writer.writerow({"run_id": 2, "dropout_p": 0.1, "best_val_mape_percent": 18.0})

        grouped = read_member_metrics(csv_path)

        self.assertEqual(grouped[1], [(0.0, 15.0), (0.2, 12.0)])
        self.assertEqual(grouped[2], [(0.1, 18.0)])


if __name__ == "__main__":
    unittest.main()
