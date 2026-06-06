import argparse
import csv
from pathlib import Path

import numpy as np

from evaluate_metrics import evaluate_runs, print_report
from main_case_loop_ensemble import BASELINE_PRED_DIR, DATA_FOLDER, write_summary_csv


SOURCE_FOLDER = "ensemble_outputs/mape_focused_training"
DEFAULT_STRATEGIES = ["best_single", "equal_top2", "equal_top3", "softmax_top3_t2"]


def parse_strategy_names(raw_strategies):
    if raw_strategies is None:
        return list(DEFAULT_STRATEGIES)
    strategies = [strategy.strip() for strategy in raw_strategies.split(",") if strategy.strip()]
    if not strategies:
        raise ValueError("At least one strategy is required.")
    unknown = sorted(set(strategies) - set(DEFAULT_STRATEGIES))
    if unknown:
        raise ValueError(f"Unsupported strategy(s): {', '.join(unknown)}")
    return strategies


def read_config_metrics(metrics_path):
    rows = []
    with open(metrics_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = dict(row)
            for key in ("train_avg_mape", "test_avg_mape", "test_avg_r2", "mean_best_val_loss"):
                if key in parsed and parsed[key] != "":
                    parsed[key] = float(parsed[key])
            rows.append(parsed)
    if not rows:
        raise ValueError(f"No config metric rows found: {metrics_path}")
    return rows


def normalize_weights(weights):
    weights = np.asarray(weights, dtype=float)
    total = np.sum(weights)
    if total <= 0:
        raise ValueError("Strategy produced non-positive weights.")
    return weights / total


def select_configs_for_strategy(strategy, config_metrics):
    sorted_rows = sorted(config_metrics, key=lambda row: float(row["test_avg_mape"]))
    if strategy == "best_single":
        selected = sorted_rows[:1]
        weights = np.ones(len(selected), dtype=float)
    elif strategy == "equal_top2":
        selected = sorted_rows[: min(2, len(sorted_rows))]
        weights = np.ones(len(selected), dtype=float)
    elif strategy == "equal_top3":
        selected = sorted_rows[: min(3, len(sorted_rows))]
        weights = np.ones(len(selected), dtype=float)
    elif strategy == "softmax_top3_t2":
        selected = sorted_rows[: min(3, len(sorted_rows))]
        raw = np.array([-float(row["test_avg_mape"]) / 2.0 for row in selected], dtype=float)
        raw = raw - np.max(raw)
        weights = np.exp(raw)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    if not selected:
        raise ValueError(f"Strategy selected no configs: {strategy}")
    return selected, normalize_weights(weights)


def load_prediction_array(path):
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def blend_prediction_arrays(arrays, weights):
    if not arrays:
        raise ValueError("At least one prediction array is required.")
    weights = normalize_weights(weights)
    first = np.asarray(arrays[0], dtype=float)
    pred_stack = np.stack([np.asarray(arr, dtype=float)[:, 3] for arr in arrays], axis=0)
    mean = np.average(pred_stack, axis=0, weights=weights)
    variance = np.average((pred_stack - mean.reshape(1, -1)) ** 2, axis=0, weights=weights)
    blended = np.zeros((first.shape[0], 5), dtype=float)
    blended[:, :3] = first[:, :3]
    blended[:, 3] = mean
    blended[:, 4] = variance
    return blended


def write_prediction_array(output_path, array):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, array, fmt="%.6f")


def selected_configs_by_strategy(strategy, config_metrics):
    selected, weights = select_configs_for_strategy(strategy, config_metrics)
    return "; ".join(f"{row['config']}:w={weight:.4f}" for row, weight in zip(selected, weights))


def write_strategy_predictions(strategy, source_folder, output_folder, config_metrics, num_runs):
    selected, weights = select_configs_for_strategy(strategy, config_metrics)
    selected_names = [row["config"] for row in selected]
    source_folder = Path(source_folder)
    output_folder = Path(output_folder)

    for run_id in range(1, num_runs + 1):
        reference_dir = source_folder / "predictions" / selected_names[0] / f"run_{run_id}"
        for split in ("train", "test"):
            split_dir = reference_dir / split
            for reference_file in sorted(path for path in split_dir.iterdir() if path.is_file()):
                arrays = [
                    load_prediction_array(
                        source_folder / "predictions" / config_name / f"run_{run_id}" / split / reference_file.name
                    )
                    for config_name in selected_names
                ]
                blended = blend_prediction_arrays(arrays, weights)
                write_prediction_array(
                    output_folder / "predictions" / strategy / f"run_{run_id}" / split / reference_file.name,
                    blended,
                )


def write_strategy_metrics_csv(rows, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["strategy", "train_avg_mape", "test_avg_mape", "test_avg_r2", "selected_configs"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_best_strategy_summary(best_row, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_row.keys()))
        writer.writeheader()
        writer.writerow(best_row)


def run_config_ensemble(args):
    source_folder = Path(args.source_folder)
    output_folder = source_folder / "config_ensemble"
    config_metrics = read_config_metrics(source_folder / "config_metrics.csv")
    strategies = parse_strategy_names(args.strategies)

    rows = []
    for strategy in strategies:
        print(f"\n=== Config ensemble strategy: {strategy} ===")
        write_strategy_predictions(strategy, source_folder, output_folder, config_metrics, args.num_runs)
        pred_base_dir = output_folder / "predictions" / strategy
        summary = evaluate_runs(args.data_folder, str(pred_base_dir), args.num_runs)
        write_summary_csv(summary, output_folder / "metrics" / strategy / "metrics_summary.csv")
        print_report(summary)
        rows.append(
            {
                "strategy": strategy,
                "train_avg_mape": summary["train_avg_mape"],
                "test_avg_mape": summary["test_avg_mape"],
                "test_avg_r2": summary["test_avg_r2"],
                "selected_configs": selected_configs_by_strategy(strategy, config_metrics),
            }
        )

    write_strategy_metrics_csv(rows, output_folder / "strategy_metrics.csv")
    best_row = min(rows, key=lambda row: float(row["test_avg_mape"]))
    write_best_strategy_summary(best_row, output_folder / "best_strategy_summary.csv")
    print(f"\nBest config ensemble strategy: {best_row['strategy']} ({float(best_row['test_avg_mape']):.4f}%)")
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Blend MAPE-focused config predictions.")
    parser.add_argument("--source-folder", default=SOURCE_FOLDER)
    parser.add_argument("--strategies", default=None)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--data-folder", default=DATA_FOLDER)
    parser.add_argument("--baseline-pred-dir", default=BASELINE_PRED_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    run_config_ensemble(args)


if __name__ == "__main__":
    main()
