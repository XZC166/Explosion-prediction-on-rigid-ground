import unittest

import numpy as np

from mape_rf_screening import (
    FORMAL_OUTPUT_FOLDER,
    OUTPUT_FOLDER,
    RFConfig,
    generate_stratified_case_folds,
    inverse_signed_log_target,
    make_mape_sample_weights,
    parse_rf_configs,
    signed_log_target,
    summarize_formal_pair_rows,
)
from mape_single_fold_screening import case_difficulty_bucket, split_manifest


class MapeRfScreeningTests(unittest.TestCase):
    def test_signed_log_target_roundtrip_preserves_signed_overpressure(self):
        values = np.array([-101325.0, -10.0, 0.0, 25.0, 100000.0])

        restored = inverse_signed_log_target(signed_log_target(values))

        np.testing.assert_allclose(restored, values, rtol=1e-12, atol=1e-8)

    def test_mape_sample_weights_clip_near_zero_dominance(self):
        y = np.array([0.0, 1000.0, 100000.0])

        weights = make_mape_sample_weights(y, clip_denominator=50000.0)

        self.assertAlmostEqual(float(np.mean(weights)), 1.0)
        self.assertAlmostEqual(weights[0], weights[1])
        self.assertGreater(weights[0], weights[2])

    def test_parse_rf_configs_accepts_semicolon_specs(self):
        configs = parse_rf_configs("fast:100:3:16:123:50000;deep:200:1:none:456:10000")

        self.assertEqual(
            configs,
            [
                RFConfig("fast", 100, 3, 16, 123, 50000.0),
                RFConfig("deep", 200, 1, None, 456, 10000.0),
            ],
        )

    def test_formal_output_folder_is_separate_from_existing_screening_outputs(self):
        self.assertNotEqual(FORMAL_OUTPUT_FOLDER, OUTPUT_FOLDER)
        self.assertIn("formal", FORMAL_OUTPUT_FOLDER)

    def test_stratified_case_folds_partition_cases_once_by_input_bucket(self):
        cases = []
        for index in range(10):
            cases.append(
                {
                    "case_file": f"low_near{index}",
                    "X": np.array([[0.0, 0.0, 0.0, 0.1, 30.0, 13.92, 0.0, 0.0, 0.0, 100000.0]]),
                    "y": np.array([[100.0]]),
                }
            )
            cases.append(
                {
                    "case_file": f"high_far{index}",
                    "X": np.array([[4000.0, 4000.0, 0.0, 5000.0, 300.0, 5129.93, 5656.0, 330.0, 5.8, 0.003]]),
                    "y": np.array([[100000.0]]),
                }
            )

        folds = generate_stratified_case_folds(cases, n_folds=5, seed=1234)

        self.assertEqual(len(folds), 5)
        tested = []
        for fold in folds:
            manifest = split_manifest(fold)
            self.assertFalse(set(manifest["train"]) & set(manifest["test"]))
            self.assertFalse(manifest["validation"])
            self.assertEqual(
                {case_difficulty_bucket(case) for case in fold.test_cases},
                {"low_charge_near", "high_charge_far"},
            )
            tested.extend(manifest["test"])
        self.assertEqual(sorted(tested), sorted(case["case_file"] for case in cases))

    def test_summarize_formal_pair_rows_reports_paired_delta_and_thresholds(self):
        rows = [
            {
                "fold": 1,
                "baseline_mape": 20.0,
                "baseline_r2": 0.50,
                "candidate_mape": 14.0,
                "candidate_r2": 0.70,
            },
            {
                "fold": 2,
                "baseline_mape": 18.0,
                "baseline_r2": 0.60,
                "candidate_mape": 16.0,
                "candidate_r2": 0.80,
            },
        ]

        summary = summarize_formal_pair_rows(rows, threshold=15.0)

        self.assertAlmostEqual(summary["baseline_avg_mape"], 19.0)
        self.assertAlmostEqual(summary["candidate_avg_mape"], 15.0)
        self.assertAlmostEqual(summary["delta_avg_mape_candidate_minus_baseline"], -4.0)
        self.assertFalse(summary["candidate_avg_below_15_percent"])
        self.assertFalse(summary["candidate_all_folds_below_15_percent"])


if __name__ == "__main__":
    unittest.main()
