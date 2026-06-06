import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
import torch

from evaluate_metrics import evaluate_runs, print_report
from main_case_loop_ensemble import (
    BASELINE_PRED_DIR,
    DATA_FOLDER,
    ImprovedMLP,
    dropout_slug,
    load_all_case_data,
    load_split_manifest,
    predict_member,
    write_baseline_comparison_csv,
    write_prediction_file,
    write_summary_csv,
)

SOURCE_FOLDER = "ensemble_outputs/dropout_p_grid"
OUTPUT_FOLDER = "ensemble_outputs/weighted_reuse"
DEFAULT_STRATEGIES = [
    "equal_all",
    "top1",
    "top2",
    "top3",
    "inv_mape_all",
    "inv_mape_top3",
    "softmax_top3_t2",
    "softmax_top3_t5",
]
DEFAULT_BEST_STRATEGY = "softmax_top3_t2"


def parse_strategy_names(raw_strategies):
    if raw_strategies is None:
        return list(DEFAULT_STRATEGIES)
    strategies = [strategy.strip() for strategy in raw_strategies.split(",") if strategy.strip()]
    if not strategies:
        raise ValueError("At least one strategy is required.")
    return strategies


def resolve_requested_strategies(raw_strategies, best_only=False):
    if raw_strategies is not None:
        return parse_strategy_names(raw_strategies)
    if best_only:
        return [DEFAULT_BEST_STRATEGY]
    return parse_strategy_names(None)


def read_member_metrics(member_metrics_path):
    grouped = {}
    with open(member_metrics_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            run_id = int(row["run_id"])
            dropout_p = float(row["dropout_p"])
            val_mape = float(row["best_val_mape_percent"])
            grouped.setdefault(run_id, []).append((dropout_p, val_mape))
    return grouped


def normalize_weights(weights):
    weights = np.array(weights, dtype=float)
    total = np.sum(weights)
    if total <= 0:
        raise ValueError("Strategy produced non-positive weights.")
    return weights / total


def select_members_for_strategy(strategy, members):
    if not members:
        raise ValueError("No members available for strategy selection.")

    members = list(members)
    sorted_members = sorted(members, key=lambda item: item[1])

    if strategy == "equal_all":
        selected = members
        weights = np.ones(len(selected), dtype=float)
    elif strategy.startswith("top"):
        k = int(strategy.replace("top", ""))
        selected = sorted_members[:k]
        weights = np.ones(len(selected), dtype=float)
    elif strategy == "inv_mape_all":
        selected = members
        weights = [1.0 / max(val_mape, 1e-8) for _, val_mape in selected]
    elif strategy == "inv_mape_top3":
        selected = sorted_members[:3]
        weights = [1.0 / max(val_mape, 1e-8) for _, val_mape in selected]
    elif strategy.startswith("softmax_top3_t"):
        temperature = float(strategy.replace("softmax_top3_t", ""))
        selected = sorted_members[:3]
        raw = np.array([-val_mape / temperature for _, val_mape in selected], dtype=float)
        raw = raw - np.max(raw)
        weights = np.exp(raw)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    if len(selected) == 0:
        raise ValueError(f"Strategy selected no members: {strategy}")
    return selected, normalize_weights(weights)


def compute_weighted_mean_and_variance(predictions, weights):
    stacked = np.stack([np.asarray(pred).reshape(-1) for pred in predictions], axis=0)
    weights = normalize_weights(weights)
    mean = np.average(stacked, axis=0, weights=weights)
    variance = np.average((stacked - mean.reshape(1, -1)) ** 2, axis=0, weights=weights)
    return mean, variance


def load_models_for_run(source_folder, run_id, members):
    models = {}
    for dropout_p, _ in members:
        model = ImprovedMLP(input_dim=10, dropout_p=dropout_p)
        model_path = (
            Path(source_folder)
            / "models"
            / f"run_{run_id}"
            / f"p_{dropout_slug(dropout_p)}.pth"
        )
        model.load_state_dict(torch.load(model_path))
        model.eval()
        models[dropout_p] = model
    return models


def write_strategy_predictions(strategy, args, all_case_data, metrics_by_run):
    source_folder = Path(args.source_folder)
    output_folder = Path(args.output_folder)
    cases_by_name = {case["case_file"]: case for case in all_case_data}

    for run_id in range(1, args.num_runs + 1):
        members = metrics_by_run[run_id]
        selected, weights = select_members_for_strategy(strategy, members)
        selected_dropout_values = [dropout_p for dropout_p, _ in selected]

        scaler = joblib.load(source_folder / "scalers" / f"scaler_X_{run_id}.pkl")
        models = load_models_for_run(source_folder, run_id, selected)
        manifest = load_split_manifest(args.baseline_pred_dir, run_id)

        for split in ("train", "test"):
            for case_name in manifest[split]:
                case = cases_by_name[case_name]
                member_predictions = [
                    predict_member(models[dropout_p], scaler, case["X"]).reshape(-1)
                    for dropout_p in selected_dropout_values
                ]
                preds_mean, preds_var = compute_weighted_mean_and_variance(member_predictions, weights)
                output_path = (
                    output_folder
                    / "predictions"
                    / strategy
                    / f"run_{run_id}"
                    / split
                    / case_name
                )
                write_prediction_file(output_path, case["raw_points"], preds_mean, preds_var)


def write_strategy_metrics_csv(rows, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "strategy",
        "train_avg_mape",
        "test_avg_mape",
        "test_avg_r2",
        "selected_by_run",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_best_strategy_summary(best_row, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(best_row.keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(best_row)


def selected_members_by_run(strategy, metrics_by_run, num_runs):
    parts = []
    for run_id in range(1, num_runs + 1):
        selected, weights = select_members_for_strategy(strategy, metrics_by_run[run_id])
        formatted = ",".join(
            f"p={dropout_p:g}:w={weight:.4f}" for (dropout_p, _), weight in zip(selected, weights)
        )
        parts.append(f"run_{run_id}[{formatted}]")
    return "; ".join(parts)


def evaluate_strategy(strategy, args):
    pred_base_dir = Path(args.output_folder) / "predictions" / strategy
    summary = evaluate_runs(args.data_folder, str(pred_base_dir), args.num_runs)
    strategy_dir = Path(args.output_folder) / "metrics" / strategy
    write_summary_csv(summary, strategy_dir / "metrics_summary.csv")

    baseline_summary = evaluate_runs(args.data_folder, args.baseline_pred_dir, args.num_runs)
    write_baseline_comparison_csv(
        baseline_summary,
        summary,
        strategy_dir / "baseline_vs_strategy.csv",
    )
    return summary


def run_weighted_inference(args):
    source_folder = Path(args.source_folder)
    metrics_by_run = read_member_metrics(source_folder / "member_metrics.csv")
    strategies = resolve_requested_strategies(args.strategies, args.best_only)
    all_case_data = load_all_case_data(args.data_folder, args.case_info_path)

    strategy_rows = []
    for strategy in strategies:
        print(f"\n=== Strategy: {strategy} ===")
        write_strategy_predictions(strategy, args, all_case_data, metrics_by_run)
        summary = evaluate_strategy(strategy, args)
        print_report(summary)
        strategy_rows.append(
            {
                "strategy": strategy,
                "train_avg_mape": summary["train_avg_mape"],
                "test_avg_mape": summary["test_avg_mape"],
                "test_avg_r2": summary["test_avg_r2"],
                "selected_by_run": selected_members_by_run(strategy, metrics_by_run, args.num_runs),
            }
        )

    output_folder = Path(args.output_folder)
    write_strategy_metrics_csv(strategy_rows, output_folder / "strategy_metrics.csv")
    best_row = min(strategy_rows, key=lambda row: float(row["test_avg_mape"]))
    write_best_strategy_summary(best_row, output_folder / "best_strategy_summary.csv")
    print(f"\nBest strategy by test MAPE: {best_row['strategy']} ({best_row['test_avg_mape']:.4f}%)")
    return strategy_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Reuse trained dropout-p members for weighted ensemble inference.")
    parser.add_argument("--strategies", default=None)
    parser.add_argument("--source-folder", default=SOURCE_FOLDER)
    parser.add_argument("--output-folder", default=OUTPUT_FOLDER)
    parser.add_argument("--best-only", action="store_true")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--data-folder", default=DATA_FOLDER)
    parser.add_argument("--case-info-path", default="data/case_info.csv")
    parser.add_argument("--baseline-pred-dir", default=BASELINE_PRED_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    run_weighted_inference(args)


if __name__ == "__main__":
    main()
