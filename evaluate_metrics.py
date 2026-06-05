import argparse
import os
import warnings

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

TRUE_DIR = "data/collect_pressure_peak"
PRED_BASE_DIR = "predictions"
DEFAULT_NUM_RUNS = 5


def evaluate_directory(true_dir, pred_dir):
    if not os.path.exists(pred_dir):
        return None

    y_true_abs_all = []
    y_pred_abs_all = []
    y_true_over_all = []
    y_pred_over_all = []

    for filename in os.listdir(pred_dir):
        true_path = os.path.join(true_dir, filename)
        pred_path = os.path.join(pred_dir, filename)

        if not os.path.exists(true_path):
            continue

        try:
            true_data = np.loadtxt(true_path)
            pred_data = np.loadtxt(pred_path)

            if len(true_data.shape) == 1:
                true_data = true_data.reshape(1, -1)
            if len(pred_data.shape) == 1:
                pred_data = pred_data.reshape(1, -1)

            true_p = true_data[:, 3]
            pred_p = pred_data[:, 3]

            y_true_abs_all.extend(true_p)
            y_pred_abs_all.extend(pred_p)
            y_true_over_all.extend(true_p - 101325)
            y_pred_over_all.extend(pred_p - 101325)
        except Exception:
            pass

    if len(y_true_abs_all) == 0:
        return None

    y_true_abs_all = np.array(y_true_abs_all)
    y_pred_abs_all = np.array(y_pred_abs_all)
    y_true_over_all = np.array(y_true_over_all)
    y_pred_over_all = np.array(y_pred_over_all)

    mape_over = np.mean(
        np.abs((y_true_over_all - y_pred_over_all) / (np.abs(y_true_over_all) + 1e-5))
    ) * 100
    mape_abs = np.mean(
        np.abs((y_true_abs_all - y_pred_abs_all) / (y_true_abs_all + 1e-5))
    ) * 100
    rmse_over = np.sqrt(mean_squared_error(y_true_over_all, y_pred_over_all))
    r2 = r2_score(y_true_over_all, y_pred_over_all)

    return {
        "mape_over": mape_over,
        "mape_abs": mape_abs,
        "rmse": rmse_over,
        "r2": r2,
        "samples": len(y_true_abs_all),
    }


def evaluate_runs(true_dir=TRUE_DIR, pred_base_dir=PRED_BASE_DIR, num_runs=DEFAULT_NUM_RUNS):
    rows = []
    run_train_mapes = []
    run_test_mapes = []
    run_test_r2s = []

    for run_id in range(1, num_runs + 1):
        run_dir = os.path.join(pred_base_dir, f"run_{run_id}")
        split_dirs = {
            "Train": os.path.join(run_dir, "train"),
            "Test": os.path.join(run_dir, "test"),
        }

        for split, split_dir in split_dirs.items():
            result = evaluate_directory(true_dir, split_dir)
            if not result:
                continue

            row = {"run_id": run_id, "split": split, **result}
            rows.append(row)

            if split == "Train":
                run_train_mapes.append(result["mape_over"])
            else:
                run_test_mapes.append(result["mape_over"])
                run_test_r2s.append(result["r2"])

    return {
        "num_runs": num_runs,
        "rows": rows,
        "train_avg_mape": float(np.mean(run_train_mapes)) if run_train_mapes else 0.0,
        "test_avg_mape": float(np.mean(run_test_mapes)) if run_test_mapes else 0.0,
        "test_avg_r2": float(np.mean(run_test_r2s)) if run_test_r2s else 0.0,
    }


def print_report(summary):
    num_runs = summary.get("num_runs", DEFAULT_NUM_RUNS)
    print(f"{'='*85}")
    print(
        f"{'Run':^5} | {'Split':^6} | {'Samples':^8} | "
        f"{'超压MAPE(%)':^14} | {'绝对压MAPE(%)':^14} | {'R2 Score':^10} | {'RMSE (Pa)':^10}"
    )
    print(f"{'-'*85}")

    for row in summary["rows"]:
        print(
            f"{row['run_id']:^5} | {row['split']:^6} | {row['samples']:^8} | "
            f"{row['mape_over']:^14.2f} | {row['mape_abs']:^14.2f} | "
            f"{row['r2']:^10.4f} | {row['rmse']:^10.1f}"
        )

    print(f"{'-'*85}")
    print(
        f"{num_runs}次交叉验证平均 => "
        f"训练集超压MAPE: {summary['train_avg_mape']:.2f}% | "
        f"测试集超压MAPE: {summary['test_avg_mape']:.2f}% | "
        f"测试集R2: {summary['test_avg_r2']:.4f}"
    )
    print(f"{'='*85}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate case-wise prediction directories.")
    parser.add_argument("--true-dir", default=TRUE_DIR, help="Ground-truth pressure directory.")
    parser.add_argument(
        "--pred-base-dir",
        default=PRED_BASE_DIR,
        help="Prediction base directory containing run_*/train and run_*/test.",
    )
    parser.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS, help="Number of runs to evaluate.")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = evaluate_runs(args.true_dir, args.pred_base_dir, args.num_runs)
    print_report(summary)


if __name__ == "__main__":
    main()
