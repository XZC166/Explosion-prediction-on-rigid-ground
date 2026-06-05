import argparse
import csv
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate_metrics import evaluate_runs, print_report

warnings.filterwarnings("ignore")

NUM_RUNS = 5
Y_SCALE_FACTOR = 1000000.0
EPOCHS = 400
BATCH_SIZE = 64
LR = 0.001
DATA_FOLDER = "data/collect_pressure_peak"
CASE_INFO_PATH = "data/case_info.csv"
BASELINE_PRED_DIR = "predictions"
OUTPUT_FOLDER = "ensemble_outputs/dropout_p_grid"
DEFAULT_DROPOUT_VALUES = "0.0,0.1,0.2,0.3,0.5"
ATM_PRESSURE = 101325.0


class ImprovedMLP(nn.Module):
    def __init__(self, input_dim, dropout_p):
        super(ImprovedMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x)


def mape_loss(pred, true):
    return torch.mean(torch.abs((pred - true) / (true + 1e-5)))


def parse_dropout_values(raw_values):
    values = []
    for raw_value in raw_values.split(","):
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        value = float(raw_value)
        if value < 0 or value >= 1:
            raise ValueError(f"Dropout p must be in [0, 1): {value}")
        values.append(value)

    if not values:
        raise ValueError("At least one dropout value is required.")
    return values


def dropout_slug(dropout_p):
    return str(dropout_p).replace(".", "_")


def build_case_features(points_data, case_row):
    blast = case_row["blast"]
    b_height = case_row["bili_height"]
    height = case_row["height"]

    case_X = []
    case_y = []
    for row in points_data:
        x, y, z, p_abs = row[0], row[1], row[2], row[3]
        p_over = p_abs - ATM_PRESSURE
        R = np.sqrt(x**2 + y**2 + z**2)
        Z = R / (blast ** (1 / 3)) if blast > 0 else R
        log_Z = np.log(Z + 1e-5)
        inv_Z = 1.0 / (Z + 1e-5)
        case_X.append([x, y, z, blast, b_height, height, R, Z, log_Z, inv_Z])
        case_y.append(p_over)

    return np.array(case_X), np.array(case_y).reshape(-1, 1)


def load_all_case_data(data_folder=DATA_FOLDER, case_info_path=CASE_INFO_PATH):
    case_info_df = pd.read_csv(case_info_path)[["id", "blast", "bili_height", "height"]]
    case_rows = {row["id"]: row for _, row in case_info_df.iterrows()}

    all_case_data = []
    for filename in sorted(os.listdir(data_folder)):
        case_path = os.path.join(data_folder, filename)
        if filename not in case_rows or not os.path.isfile(case_path):
            continue

        try:
            points_data = np.loadtxt(case_path)
            if len(points_data.shape) == 1:
                points_data = points_data.reshape(1, -1)
        except Exception:
            continue

        case_X, case_y = build_case_features(points_data, case_rows[filename])
        if len(case_X) == 0:
            continue

        all_case_data.append(
            {
                "case_file": filename,
                "X": case_X,
                "y": case_y,
                "raw_points": points_data,
            }
        )

    return all_case_data


def list_split_files(split_dir):
    split_path = Path(split_dir)
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split directory: {split_path}")
    return sorted(path.name for path in split_path.iterdir() if path.is_file() and not path.name.startswith("."))


def load_split_manifest(pred_base_dir, run_id):
    run_dir = Path(pred_base_dir) / f"run_{run_id}"
    return {
        "train": list_split_files(run_dir / "train"),
        "test": list_split_files(run_dir / "test"),
    }


def cases_from_manifest(all_case_data, manifest):
    cases_by_name = {case["case_file"]: case for case in all_case_data}
    missing = sorted((set(manifest["train"]) | set(manifest["test"])) - set(cases_by_name))
    if missing:
        raise ValueError(f"Split manifest references missing data files: {missing}")

    train_cases = [dict(cases_by_name[name], split="train") for name in manifest["train"]]
    test_cases = [dict(cases_by_name[name], split="test") for name in manifest["test"]]
    return train_cases, test_cases


def make_loaders(train_cases, test_cases, batch_size, seed):
    X_train = np.vstack([case["X"] for case in train_cases])
    y_train = np.vstack([case["y"] for case in train_cases])
    X_test = np.vstack([case["X"] for case in test_cases])
    y_test = np.vstack([case["y"] for case in test_cases])

    scaler_X = StandardScaler()
    X_train_norm = scaler_X.fit_transform(X_train)
    X_test_norm = scaler_X.transform(X_test)

    y_train_scaled = y_train / Y_SCALE_FACTOR
    y_test_scaled = y_test / Y_SCALE_FACTOR

    train_dataset = TensorDataset(
        torch.tensor(X_train_norm, dtype=torch.float32),
        torch.tensor(y_train_scaled, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test_norm, dtype=torch.float32),
        torch.tensor(y_test_scaled, dtype=torch.float32),
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return scaler_X, train_loader, test_loader, len(X_train), len(X_test)


def train_member(train_loader, test_loader, dropout_p, model_save_path, epochs, lr, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = ImprovedMLP(input_dim=10, dropout_p=dropout_p)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = mape_loss(pred, batch_y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    val_loss += mape_loss(model(batch_X), batch_y).item()
            val_loss /= len(test_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_save_path)

    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    return model, best_val_loss


def predict_member(model, scaler_X, case_X):
    case_X_norm = scaler_X.transform(case_X)
    case_X_tensor = torch.tensor(case_X_norm, dtype=torch.float32)
    with torch.no_grad():
        preds_scaled = model(case_X_tensor).numpy()
    return (preds_scaled * Y_SCALE_FACTOR) + ATM_PRESSURE


def ensemble_member_predictions(member_predictions):
    stacked = np.stack([np.asarray(pred).reshape(-1) for pred in member_predictions], axis=0)
    return np.mean(stacked, axis=0), np.var(stacked, axis=0)


def write_prediction_file(output_path, raw_points, preds_mean, preds_var):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(len(raw_points)):
            x = raw_points[i, 0]
            y = raw_points[i, 1]
            z = raw_points[i, 2]
            f.write(f"{x} {y} {z} {preds_mean[i]:.4f} {preds_var[i]:.4f}\n")


def write_summary_csv(summary, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_id", "split", "samples", "mape_over", "mape_abs", "r2", "rmse"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow({key: row[key] for key in fieldnames})


def write_member_metrics_csv(member_rows, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_id", "dropout_p", "best_val_mape_percent"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(member_rows)


def write_baseline_comparison_csv(baseline_summary, ensemble_summary, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    baseline_rows = {(row["run_id"], row["split"]): row for row in baseline_summary["rows"]}
    ensemble_rows = {(row["run_id"], row["split"]): row for row in ensemble_summary["rows"]}
    fieldnames = [
        "run_id",
        "split",
        "baseline_mape_over",
        "ensemble_mape_over",
        "delta_mape_over",
        "baseline_r2",
        "ensemble_r2",
        "delta_r2",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(ensemble_rows):
            if key not in baseline_rows:
                continue
            baseline = baseline_rows[key]
            ensemble = ensemble_rows[key]
            writer.writerow(
                {
                    "run_id": key[0],
                    "split": key[1],
                    "baseline_mape_over": baseline["mape_over"],
                    "ensemble_mape_over": ensemble["mape_over"],
                    "delta_mape_over": ensemble["mape_over"] - baseline["mape_over"],
                    "baseline_r2": baseline["r2"],
                    "ensemble_r2": ensemble["r2"],
                    "delta_r2": ensemble["r2"] - baseline["r2"],
                }
            )


def run_fold(run_id, all_case_data, dropout_values, args):
    print(f"\n{'='*40}")
    print(f"          开始第 {run_id} 折 dropout 集成实验          ")
    print(f"{'='*40}")

    seed = 42 + run_id
    manifest = load_split_manifest(args.baseline_pred_dir, run_id)
    train_cases, test_cases = cases_from_manifest(all_case_data, manifest)
    all_cases = train_cases + test_cases

    scaler_X, train_loader, test_loader, train_points, test_points = make_loaders(
        train_cases, test_cases, args.batch_size, seed
    )

    output_root = Path(args.output_folder)
    model_dir = output_root / "models" / f"run_{run_id}"
    scaler_dir = output_root / "scalers"
    prediction_run_dir = output_root / "predictions" / f"run_{run_id}"
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_X, scaler_dir / f"scaler_X_{run_id}.pkl")

    print(f"|-- split复用完成：训练算例 {len(train_cases)} 个，测试算例 {len(test_cases)} 个 --|")
    print(f"|-- 测点数量：训练 {train_points}，测试 {test_points} --|")

    member_models = []
    member_metrics = []
    for dropout_p in dropout_values:
        model_save_path = model_dir / f"p_{dropout_slug(dropout_p)}.pth"
        print(f"-> 训练成员 p={dropout_p:g}，保存到 {model_save_path}")
        model, best_val_loss = train_member(
            train_loader=train_loader,
            test_loader=test_loader,
            dropout_p=dropout_p,
            model_save_path=model_save_path,
            epochs=args.epochs,
            lr=args.lr,
            seed=seed,
        )
        member_models.append(model)
        member_metrics.append(
            {
                "run_id": run_id,
                "dropout_p": dropout_p,
                "best_val_mape_percent": best_val_loss * 100,
            }
        )
        print(f"   best val MAPE: {best_val_loss * 100:.2f}%")

    for case in all_cases:
        member_predictions = [predict_member(model, scaler_X, case["X"]) for model in member_models]
        preds_mean, preds_var = ensemble_member_predictions(member_predictions)
        target_dir = prediction_run_dir / case["split"]
        write_prediction_file(target_dir / case["case_file"], case["raw_points"], preds_mean, preds_var)

    expected_train = sorted(path.name for path in (prediction_run_dir / "train").iterdir() if path.is_file())
    expected_test = sorted(path.name for path in (prediction_run_dir / "test").iterdir() if path.is_file())
    if expected_train != manifest["train"] or expected_test != manifest["test"]:
        raise RuntimeError(f"Output split files do not match baseline manifest for run {run_id}.")

    print(f"-> 第 {run_id} 折集成预测已保存。\n")
    return member_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train a fair dropout-p grid ensemble for case-wise prediction.")
    parser.add_argument("--num-runs", type=int, default=NUM_RUNS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--dropout-values", default=DEFAULT_DROPOUT_VALUES)
    parser.add_argument("--output-folder", default=OUTPUT_FOLDER)
    parser.add_argument("--data-folder", default=DATA_FOLDER)
    parser.add_argument("--case-info-path", default=CASE_INFO_PATH)
    parser.add_argument("--baseline-pred-dir", default=BASELINE_PRED_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    return parser.parse_args()


def main():
    args = parse_args()
    dropout_values = parse_dropout_values(args.dropout_values)
    all_case_data = load_all_case_data(args.data_folder, args.case_info_path)
    if not all_case_data:
        raise RuntimeError(f"No valid case data loaded from {args.data_folder}")

    print(f"Loaded {len(all_case_data)} cases.")
    print(f"Dropout grid: {dropout_values}")
    print(f"Output folder: {args.output_folder}")

    all_member_metrics = []
    for run_id in range(1, args.num_runs + 1):
        all_member_metrics.extend(run_fold(run_id, all_case_data, dropout_values, args))

    output_root = Path(args.output_folder)
    ensemble_summary = evaluate_runs(
        true_dir=args.data_folder,
        pred_base_dir=str(output_root / "predictions"),
        num_runs=args.num_runs,
    )
    baseline_summary = evaluate_runs(
        true_dir=args.data_folder,
        pred_base_dir=args.baseline_pred_dir,
        num_runs=args.num_runs,
    )

    write_summary_csv(ensemble_summary, output_root / "metrics_summary.csv")
    write_baseline_comparison_csv(baseline_summary, ensemble_summary, output_root / "baseline_vs_ensemble.csv")
    write_member_metrics_csv(all_member_metrics, output_root / "member_metrics.csv")

    print("\n集成模型评估结果：")
    print_report(ensemble_summary)
    print(f"Metrics saved to: {output_root / 'metrics_summary.csv'}")
    print(f"Baseline comparison saved to: {output_root / 'baseline_vs_ensemble.csv'}")
    print(f"Member metrics saved to: {output_root / 'member_metrics.csv'}")


if __name__ == "__main__":
    main()
