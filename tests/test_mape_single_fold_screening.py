import unittest
import uuid
from pathlib import Path

import numpy as np

from mape_single_fold_screening import (
    DEFAULT_SCREENING_STRATEGIES,
    case_difficulty_bucket,
    generate_case_split,
    generate_stratified_case_split,
    make_screening_model,
    select_best_strategy_by_validation,
    select_members_for_strategy,
    split_manifest,
    write_strategy_predictions,
)


TEST_WORKSPACE = Path.cwd() / "_test_workspace"


def reset_test_dir(name):
    root = TEST_WORKSPACE / f"{name}_{uuid.uuid4().hex}"
    root.mkdir(parents=True)
    return root


class SingleFoldMapeScreeningTests(unittest.TestCase):
    def test_generate_case_split_is_reproducible_and_disjoint(self):
        cases = [{"case_file": f"value{i}"} for i in range(1, 11)]

        first = generate_case_split(cases, test_fraction=0.2, validation_fraction=0.25, seed=7)
        second = generate_case_split(cases, test_fraction=0.2, validation_fraction=0.25, seed=7)

        self.assertEqual(split_manifest(first), split_manifest(second))
        train = set(split_manifest(first)["train"])
        validation = set(split_manifest(first)["validation"])
        test = set(split_manifest(first)["test"])
        self.assertTrue(train)
        self.assertTrue(validation)
        self.assertTrue(test)
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(train | validation | test, {f"value{i}" for i in range(1, 11)})

    def test_stratified_split_keeps_input_feature_buckets_in_validation_and_test(self):
        cases = []
        for index in range(4):
            cases.append(
                {
                    "case_file": f"low_near{index}",
                    "X": np.array([[0.0, 0.0, 0.0, 0.1, 30.0, 13.92, 0.0, 0.0, 0.0, 100000.0]]),
                    "y": np.array([[-100.0]]),
                }
            )
            cases.append(
                {
                    "case_file": f"low_far{index}",
                    "X": np.array([[3000.0, 3000.0, 0.0, 0.1, 300.0, 139.25, 4242.0, 9137.0, 9.1, 0.0001]]),
                    "y": np.array([[1000.0]]),
                }
            )
            cases.append(
                {
                    "case_file": f"high_far{index}",
                    "X": np.array([[4000.0, 4000.0, 0.0, 5000.0, 300.0, 5129.93, 5656.0, 330.0, 5.8, 0.003]]),
                    "y": np.array([[100000.0]]),
                }
            )

        first = generate_stratified_case_split(cases, test_fraction=0.25, validation_fraction=0.34, seed=11)
        second = generate_stratified_case_split(cases, test_fraction=0.25, validation_fraction=0.34, seed=11)

        self.assertEqual(split_manifest(first), split_manifest(second))
        for split_name in ("train", "validation", "test"):
            split_cases = getattr(first, f"{split_name}_cases")
            self.assertEqual(
                {case_difficulty_bucket(case) for case in split_cases},
                {"low_charge_near", "low_charge_far", "high_charge_far"},
            )

    def test_case_difficulty_bucket_uses_only_known_input_features(self):
        near_low_charge = {
            "case_file": "same_input_a",
            "X": np.array([[0.0, 0.0, 0.0, 0.1, 30.0, 13.92, 0.0, 0.0, 0.0, 100000.0]]),
            "y": np.array([[-101325.0], [500000.0]]),
        }
        same_input_different_truth = {
            **near_low_charge,
            "case_file": "same_input_b",
            "y": np.array([[100.0], [100.0]]),
        }
        far_high_charge = {
            "case_file": "far_high_charge",
            "X": np.array([[4000.0, 4000.0, 0.0, 5000.0, 300.0, 5129.93, 5656.0, 330.0, 5.8, 0.003]]),
            "y": np.array([[100.0]]),
        }

        self.assertEqual(case_difficulty_bucket(near_low_charge), case_difficulty_bucket(same_input_different_truth))
        self.assertNotEqual(case_difficulty_bucket(near_low_charge), case_difficulty_bucket(far_high_charge))

    def test_best_strategy_uses_validation_mape_not_test_mape(self):
        rows = [
            {"strategy": "validation_winner", "validation_avg_mape": 11.0, "test_avg_mape": 99.0},
            {"strategy": "test_winner", "validation_avg_mape": 22.0, "test_avg_mape": 1.0},
        ]

        best = select_best_strategy_by_validation(rows)

        self.assertEqual(best["strategy"], "validation_winner")

    def test_default_strategies_include_validation_weight_optimizer(self):
        self.assertIn("validation_weight_opt", DEFAULT_SCREENING_STRATEGIES)
        self.assertIn("softmax_top3_t2", DEFAULT_SCREENING_STRATEGIES)

    def test_screening_model_can_disable_positive_output_constraint(self):
        positive_model = make_screening_model(dropout_p=0.1, allow_negative_output=False)
        signed_model = make_screening_model(dropout_p=0.1, allow_negative_output=True)

        self.assertEqual(positive_model.net[-1].__class__.__name__, "Softplus")
        self.assertNotEqual(signed_model.net[-1].__class__.__name__, "Softplus")

    def test_validation_weight_optimizer_uses_validation_predictions(self):
        root = reset_test_dir("single_fold_weight_opt")
        true_dir = root / "truth"
        rows = np.array([[0.0, 0.0, 0.0, 201325.0], [1.0, 0.0, 0.0, 301325.0]])
        true_dir.mkdir(parents=True)
        np.savetxt(true_dir / "value1", rows)
        for member_id, pred_abs in (("member_a", [201325.0, 301325.0]), ("member_b", [401325.0, 101325.0])):
            pred_path = root / "member_predictions" / member_id / "run_1" / "validation" / "value1"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                pred_path,
                np.array([[0.0, 0.0, 0.0, pred_abs[0], 0.0], [1.0, 0.0, 0.0, pred_abs[1], 0.0]]),
            )
        members = [
            {"member_id": "member_a", "validation_mape_percent": 20.0},
            {"member_id": "member_b", "validation_mape_percent": 21.0},
        ]

        selected, weights = select_members_for_strategy(
            "validation_weight_opt",
            members,
            output_folder=root,
            true_dir=true_dir,
            weight_opt_top_k=2,
            weight_opt_l2=0.0,
        )

        self.assertEqual([row["member_id"] for row in selected], ["member_a", "member_b"])
        self.assertGreater(weights[0], 0.95)

    def test_write_strategy_predictions_can_write_validation_without_test(self):
        root = reset_test_dir("single_fold_defer_test")
        for member_id, pred in (("member_a", 10.0), ("member_b", 20.0)):
            for split in ("validation", "test"):
                pred_path = root / "member_predictions" / member_id / "run_1" / split / "value1"
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(pred_path, np.array([[1.0, 2.0, 3.0, pred, 0.0]]))
        members = [{"member_id": "member_a"}, {"member_id": "member_b"}]

        write_strategy_predictions("top2", root, members, np.array([0.5, 0.5]), splits=("validation",))

        self.assertTrue((root / "predictions" / "top2" / "run_1" / "validation" / "value1").exists())
        self.assertFalse((root / "predictions" / "top2" / "run_1" / "test" / "value1").exists())


if __name__ == "__main__":
    unittest.main()
