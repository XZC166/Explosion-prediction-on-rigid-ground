import unittest
import uuid
from pathlib import Path

import numpy as np

from mape_diverse_reuse import (
    DEFAULT_MEMBER_CONFIGS,
    LEGACY_OUTPUT_FOLDER,
    MemberConfig,
    OUTPUT_FOLDER,
    choose_best_strategy_by_validation,
    fold_prediction_is_complete,
    merge_member_metric_rows,
    parse_member_configs,
    select_members_for_strategy,
    split_train_validation_cases,
    write_strategy_predictions,
)

TEST_WORKSPACE = Path.cwd() / "_test_workspace"


def reset_test_dir(name):
    root = TEST_WORKSPACE / f"{name}_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    return root


class MapeDiverseReuseTests(unittest.TestCase):
    def test_default_output_folder_keeps_strict_results_separate_from_fast_trial(self):
        self.assertNotEqual(Path(OUTPUT_FOLDER), Path(LEGACY_OUTPUT_FOLDER))
        self.assertEqual(Path(OUTPUT_FOLDER).name, "mape_diverse_reuse_strict")
        self.assertEqual(Path(LEGACY_OUTPUT_FOLDER).name, "mape_diverse_reuse")

    def test_default_member_configs_are_eight_unique_mape_loss_variants(self):
        names = [config.name for config in DEFAULT_MEMBER_CONFIGS]

        self.assertEqual(len(DEFAULT_MEMBER_CONFIGS), 8)
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(0.0 <= config.dropout_p < 1.0 for config in DEFAULT_MEMBER_CONFIGS))
        self.assertTrue(all(config.lr > 0.0 for config in DEFAULT_MEMBER_CONFIGS))
        self.assertTrue(all(config.weight_decay >= 0.0 for config in DEFAULT_MEMBER_CONFIGS))

    def test_split_train_validation_cases_uses_only_outer_train_cases(self):
        train_cases = [
            {"case_file": f"value{i}", "split": "train", "X": np.zeros((1, 1)), "y": np.zeros((1, 1))}
            for i in range(1, 7)
        ]

        inner_train, validation = split_train_validation_cases(
            train_cases,
            validation_fraction=0.34,
            seed=123,
        )

        train_names = {case["case_file"] for case in inner_train}
        validation_names = {case["case_file"] for case in validation}
        original_names = {case["case_file"] for case in train_cases}
        self.assertTrue(train_names)
        self.assertTrue(validation_names)
        self.assertEqual(train_names | validation_names, original_names)
        self.assertFalse(train_names & validation_names)
        self.assertTrue(all(case["split"] == "train" for case in inner_train))
        self.assertTrue(all(case["split"] == "validation" for case in validation))

        second_train, second_validation = split_train_validation_cases(
            train_cases,
            validation_fraction=0.34,
            seed=123,
        )
        self.assertEqual(
            [case["case_file"] for case in validation],
            [case["case_file"] for case in second_validation],
        )
        self.assertEqual(
            [case["case_file"] for case in inner_train],
            [case["case_file"] for case in second_train],
        )

    def test_parse_member_configs_accepts_compact_semicolon_spec(self):
        configs = parse_member_configs(
            "p0_lr1e-3_wd0_s0:0.0:0.001:0:0;"
            "p1_lr5e-4_wd1e-6_s2:0.1:0.0005:0.000001:2"
        )

        self.assertEqual(
            configs,
            [
                MemberConfig("p0_lr1e-3_wd0_s0", 0.0, 0.001, 0.0, 0),
                MemberConfig("p1_lr5e-4_wd1e-6_s2", 0.1, 0.0005, 0.000001, 2),
            ],
        )

    def test_select_members_for_top_k_uses_lowest_validation_mape(self):
        members = [
            {"member_id": "slow", "best_val_mape_percent": 20.0},
            {"member_id": "best", "best_val_mape_percent": 10.0},
            {"member_id": "second", "best_val_mape_percent": 12.0},
        ]

        selected, weights = select_members_for_strategy("top2", members)

        self.assertEqual([member["member_id"] for member in selected], ["best", "second"])
        np.testing.assert_allclose(weights, [0.5, 0.5])

    def test_softmax_top3_weights_prefer_lower_mape_and_sum_to_one(self):
        members = [
            {"member_id": "a", "best_val_mape_percent": 10.0},
            {"member_id": "b", "best_val_mape_percent": 20.0},
            {"member_id": "c", "best_val_mape_percent": 30.0},
            {"member_id": "d", "best_val_mape_percent": 40.0},
        ]

        selected, weights = select_members_for_strategy("softmax_top3_t5", members)

        self.assertEqual([member["member_id"] for member in selected], ["a", "b", "c"])
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=7)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

    def test_fold_prediction_is_complete_requires_train_validation_and_test_files(self):
        root = reset_test_dir("diverse_fold_complete")
        run_dir = root / "member_predictions" / "member_a" / "run_1"
        manifest = {"train": ["value1", "value2"], "validation": ["value4"], "test": ["value3"]}
        for split, files in manifest.items():
            for filename in files:
                path = run_dir / split / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

        self.assertTrue(fold_prediction_is_complete(root, "member_a", 1, manifest))

        (run_dir / "test" / "extra").touch()
        self.assertFalse(fold_prediction_is_complete(root, "member_a", 1, manifest))

    def test_choose_best_strategy_uses_validation_not_test(self):
        rows = [
            {"strategy": "validation_winner", "validation_avg_mape": 10.0, "test_avg_mape": 99.0},
            {"strategy": "test_winner", "validation_avg_mape": 20.0, "test_avg_mape": 1.0},
        ]

        best = choose_best_strategy_by_validation(rows)

        self.assertEqual(best["strategy"], "validation_winner")

    def test_write_strategy_predictions_can_defer_test_outputs_until_final_selection(self):
        root = reset_test_dir("diverse_defer_test")
        for member_id, pred in (("member_a", 10.0), ("member_b", 20.0)):
            for split in ("validation", "test"):
                path = root / "member_predictions" / member_id / "run_1" / split / "value1"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(path, np.array([[1.0, 2.0, 3.0, pred, 0.0]]))
        metrics_by_run = {
            1: [
                {"member_id": "member_a", "validation_mape_percent": 10.0},
                {"member_id": "member_b", "validation_mape_percent": 20.0},
            ]
        }

        write_strategy_predictions("top1", root, metrics_by_run, num_runs=1, splits=("validation",))

        self.assertTrue((root / "predictions" / "top1" / "run_1" / "validation" / "value1").exists())
        self.assertFalse((root / "predictions" / "top1" / "run_1" / "test" / "value1").exists())

    def test_merge_member_metric_rows_replaces_existing_fold_row(self):
        existing = [
            {"run_id": 1, "member_id": "member_a", "best_val_mape_percent": 20.0},
            {"run_id": 2, "member_id": "member_a", "best_val_mape_percent": 30.0},
        ]
        new_rows = [
            {"run_id": 1, "member_id": "member_a", "best_val_mape_percent": 10.0},
            {"run_id": 1, "member_id": "member_b", "best_val_mape_percent": 40.0},
        ]

        merged = merge_member_metric_rows(existing, new_rows)

        self.assertEqual(
            [(row["member_id"], row["run_id"], row["best_val_mape_percent"]) for row in merged],
            [("member_a", 1, 10.0), ("member_a", 2, 30.0), ("member_b", 1, 40.0)],
        )


if __name__ == "__main__":
    unittest.main()
