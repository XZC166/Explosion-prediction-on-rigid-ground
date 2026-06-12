# Single-Fold MAPE Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a strict single-fold screening experiment that optimizes MAPE without test leakage.

**Architecture:** Add a focused script that reuses existing data loading, MLP, prediction writing, and split evaluation helpers from `main_case_loop_ensemble.py` and `mape_diverse_reuse.py`. The script owns random split manifest generation, baseline training, candidate member training, validation-only strategy selection, final test evaluation, and summary output.

**Tech Stack:** Python, NumPy, PyTorch, scikit-learn, unittest, existing project evaluation helpers.

---

### Task 1: Add Single-Fold Behavior Tests

**Files:**
- Create: `tests/test_mape_single_fold_screening.py`

- [ ] **Step 1: Write failing tests**

Add tests that import planned functions from `mape_single_fold_screening.py`:

```python
import unittest
from pathlib import Path

import numpy as np

from mape_single_fold_screening import (
    DEFAULT_SCREENING_STRATEGIES,
    generate_case_split,
    select_best_strategy_by_validation,
    split_manifest,
)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_mape_single_fold_screening -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'mape_single_fold_screening'`.

### Task 2: Implement Screening Helpers

**Files:**
- Create: `mape_single_fold_screening.py`

- [ ] **Step 1: Add minimal helper implementation**

Implement dataclasses and helpers:

```python
from dataclasses import dataclass

import numpy as np

DEFAULT_SCREENING_STRATEGIES = [
    "top2",
    "top3",
    "equal_top3",
    "inv_mape_top3",
    "softmax_top3_t2",
    "softmax_top3_t5",
    "validation_weight_opt",
]


@dataclass(frozen=True)
class CaseSplit:
    train_cases: list
    validation_cases: list
    test_cases: list


def generate_case_split(all_cases, test_fraction, validation_fraction, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(all_cases))
    rng.shuffle(indices)
    test_count = max(1, min(len(all_cases) - 2, int(round(len(all_cases) * test_fraction))))
    remaining = indices[test_count:]
    validation_count = max(1, min(len(remaining) - 1, int(round(len(remaining) * validation_fraction))))
    test_indices = set(int(index) for index in indices[:test_count])
    validation_indices = set(int(index) for index in remaining[:validation_count])
    train_cases = []
    validation_cases = []
    test_cases = []
    for index, case in enumerate(all_cases):
        if index in test_indices:
            test_cases.append(dict(case, split="test"))
        elif index in validation_indices:
            validation_cases.append(dict(case, split="validation"))
        else:
            train_cases.append(dict(case, split="train"))
    return CaseSplit(train_cases, validation_cases, test_cases)


def split_manifest(case_split):
    return {
        "train": [case["case_file"] for case in case_split.train_cases],
        "validation": [case["case_file"] for case in case_split.validation_cases],
        "test": [case["case_file"] for case in case_split.test_cases],
    }


def select_best_strategy_by_validation(rows):
    return min(rows, key=lambda row: float(row["validation_avg_mape"]))
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m unittest tests.test_mape_single_fold_screening -v`

Expected: PASS.

### Task 3: Add Training, Prediction, and Strategy Pipeline

**Files:**
- Modify: `mape_single_fold_screening.py`

- [ ] **Step 1: Add CLI and reuse existing project helpers**

Extend the script with:

```python
from main_case_loop_ensemble import load_all_case_data, make_loaders, ImprovedMLP, predict_member, write_prediction_file
from mape_diverse_reuse import (
    DEFAULT_MEMBER_CONFIGS,
    MemberConfig,
    blend_prediction_arrays,
    evaluate_prediction_splits,
    parse_member_configs,
    read_member_metric_rows,
    select_members_for_strategy,
    write_prediction_array,
)
```

Add baseline/member training functions that train on inner train and checkpoint on validation, then write train/validation/test predictions.

- [ ] **Step 2: Add validation-only strategy reuse**

For each strategy, write only validation predictions, evaluate validation MAPE, choose the best strategy by validation MAPE, then write test predictions only for the frozen best strategy.

- [ ] **Step 3: Add output summaries**

Write `split_manifest.csv`, `baseline_summary.csv`, `member_metrics.csv`, `strategy_metrics.csv`, `best_strategy_summary.csv`, and `candidate_vs_baseline.csv`.

### Task 4: Run Verification

**Files:**
- Test: `tests/test_mape_single_fold_screening.py`
- Test: existing `tests/*.py`

- [ ] **Step 1: Run targeted tests**

Run: `python -m unittest tests.test_mape_single_fold_screening -v`

Expected: PASS.

- [ ] **Step 2: Run full unit suite**

Run: `python -m unittest discover -s tests`

Expected: PASS.

- [ ] **Step 3: Run smoke screening**

Run: `python mape_single_fold_screening.py --output-folder ensemble_outputs/mape_single_fold_screening_smoke --epochs 2 --eval-every 1 --member-configs "p0_smoke:0.0:0.001:0:0;p01_smoke:0.1:0.001:0:1" --strategies "top2,softmax_top3_t2,validation_weight_opt"`

Expected: completes, writes manifest and summaries, and reports baseline and candidate test MAPE.

- [ ] **Step 4: Run full screening**

Run: `python mape_single_fold_screening.py --output-folder ensemble_outputs/mape_single_fold_screening --epochs 400 --eval-every 20`

Expected: completes with `best_strategy_summary.csv` and `candidate_vs_baseline.csv`. The result is a screening result, not a final five-fold formal result.
