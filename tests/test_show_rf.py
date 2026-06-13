import unittest
from pathlib import Path

import numpy as np

from show_rf import (
    build_point_dataframe,
    discover_prediction_dirs,
    filter_case_points,
    load_rf_predictions,
    single_case_plot_series,
)


class ShowRfTests(unittest.TestCase):
    def test_discover_prediction_dirs_returns_sorted_split_dirs(self):
        root = Path("tests/fixtures/show_rf")
        pred_base = root / "candidate"

        dirs = discover_prediction_dirs(pred_base, split="test")

        self.assertEqual([path.parent.name for path in dirs], ["run_1", "run_2", "run_10"])

    def test_load_rf_predictions_matches_true_files_and_subtracts_atmosphere(self):
        root = Path("tests/fixtures/show_rf")
        pred_base = root / "candidate"
        true_dir = root / "truth"

        loaded = load_rf_predictions(pred_base, true_dir, split="test", atm_pressure=101325.0)

        np.testing.assert_allclose(loaded.y_pred, np.array([1000.0, 2000.0, 100.0, 200.0]))
        np.testing.assert_allclose(loaded.y_true, np.array([500.0, 3000.0, 300.0, 400.0]))
        self.assertEqual(int(loaded.case_results.iloc[0]["fold"]), 1)
        self.assertEqual(loaded.case_results.iloc[0]["case_file"], "value1")
        self.assertEqual(int(loaded.case_results.iloc[0]["n_samples"]), 2)

    def test_build_point_dataframe_adds_distance_and_kpa_columns(self):
        root = Path("tests/fixtures/show_rf")
        pred_base = root / "candidate"
        true_dir = root / "truth"

        df = build_point_dataframe(pred_base, true_dir, split="test", atm_pressure=101325.0)

        self.assertEqual(list(df.columns), ["fold", "case_file", "x", "y", "z", "distance", "true_kpa", "pred_kpa"])
        self.assertEqual(len(df), 4)
        np.testing.assert_allclose(df["distance"].to_numpy()[:2], np.array([0.0, 2 ** 0.5]))
        np.testing.assert_allclose(df.sort_values("true_kpa")["true_kpa"].to_numpy(), np.array([0.3, 0.4, 0.5, 3.0]))

    def test_filter_case_points_accepts_case_numbers_and_value_names(self):
        root = Path("tests/fixtures/show_rf")
        df = build_point_dataframe(root / "candidate", root / "truth", split="test", atm_pressure=101325.0)

        filtered = filter_case_points(df, ["1", "value10"])

        self.assertEqual(sorted(filtered["case_file"].unique()), ["value1", "value10"])
        self.assertEqual(len(filtered), 3)

    def test_single_case_plot_series_converts_kpa_to_pa(self):
        root = Path("tests/fixtures/show_rf")
        df = build_point_dataframe(root / "candidate", root / "truth", split="test", atm_pressure=101325.0)
        case_df = filter_case_points(df, ["1"])

        series = single_case_plot_series(case_df, y_scale=1000.0)

        np.testing.assert_allclose(series["true"], np.array([500.0, 3000.0]))
        np.testing.assert_allclose(series["pred"], np.array([1000.0, 2000.0]))


if __name__ == "__main__":
    unittest.main()
