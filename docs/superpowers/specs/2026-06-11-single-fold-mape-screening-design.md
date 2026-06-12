# Single-Fold MAPE Screening Design

## Goal

Build a strict single-fold screening path for MAPE-only optimization. The screening run must compare a baseline model and a candidate ensemble on the same random case-wise split, use only inner validation for checkpoint/member/strategy selection, and evaluate the outer test split once after the strategy is frozen.

## Experiment Rules

- Generate and save one random case-wise split manifest with train, validation, and test case lists.
- Train the baseline and all candidate members on the same inner-train cases.
- Use validation only for checkpoint selection, member ranking, ensemble weighting, and strategy selection.
- Do not use outer test predictions while selecting members or strategies.
- Evaluate test only for the validation-selected baseline and candidate strategy.
- Store outputs under a new experiment directory so existing historical outputs stay intact.

## Candidate Approach

The first candidate keeps the current 10 physical features and MAPE loss, then trains a small diverse member pool across dropout, learning rate, weight decay, and seed. Strategy candidates are limited to `top2`, `top3`, `equal_top3`, `inv_mape_top3`, `softmax_top3_t2`, `softmax_top3_t5`, and a constrained validation-only nonnegative weight optimizer.

## Files

- `mape_single_fold_screening.py`: new command-line script for strict single-fold screening.
- `tests/test_mape_single_fold_screening.py`: behavior tests for manifests, strategy selection, and test deferral.
- `docs/superpowers/plans/2026-06-11-single-fold-mape-screening.md`: implementation plan.

## Verification

Run unit tests first, then run a short smoke experiment with low epochs to verify file layout. A full screening run can then use 400 epochs and the default member pool. Success is measured by candidate test MAPE relative to the same-split baseline test MAPE, with the target of moving below 15%.
