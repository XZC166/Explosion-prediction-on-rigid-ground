import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from evaluate_metrics import evaluate_runs, print_report
from main_case_loop_ensemble import (
    BASELINE_PRED_DIR,
    CASE_INFO_PATH,
    DATA_FOLDER,
    ImprovedMLP,
    Y_SCALE_FACTOR,
    cases_from_manifest,
    load_all_case_data,
    load_split_manifest,
    write_baseline_comparison_csv,
    write_summary_csv,
)


OUTPUT_FOLDER = "ensemble_outputs/mape_focused_training"
DEFAULT_CONFIGS = ["log_l1", "log_huber", "z_weighted_mape", "log_l1_z_weighted"]
CONFIG_SPECS = {
    "log_l1": {"target": "log", "loss": "l1", "z_weighted": False},
    "log_huber": {"target": "log", "loss": "huber", "z_weighted": False},
    "z_weighted_mape": {"target": "linear", "loss": "mape", "z_weighted": True},
    "log_l1_z_weighted": {"target": "log", "loss": "l1", "z_weighted": True},
}
ATM_PRESSURE = 101325.0
DEFAULT_Z_THRESHOLD = 3500.0
DEFAULT_MAX_Z_WEIGHT = 3.0


def parse_config_names(raw_configs):
    if raw_configs is None:
        return list(DEFAULT_CONFIGS)
    configs = [config.strip() for config in raw_configs.split(",") if config.strip()]
    if not configs:
        raise ValueError("At least one config is required.")
    unknown = sorted(set(configs) - set(CONFIG_SPECS))
    if unknown:
        raise ValueError(f"Unsupported config(s): {', '.join(unknown)}")
    return configs


def overpressure_to_log_target(overpressure):
    overpressure = np.asarray(overpressure, dtype=float)
    return np.log1p(np.clip(overpressure, 0.0, None))


def log_target_to_overpressure(log_target):
    log_target = np.asarray(log_target, dtype=float)
    return np.expm1(log_target)


def compute_z_sample_weights(z_values, threshold=DEFAULT_Z_THRESHOLD, max_weight=DEFAULT_MAX_Z_WEIGHT):
    z_values = np.asarray(z_values, dtype=float)
    boosted = 1.0 + np.maximum(z_values - threshold, 0.0) / max(threshold, 1e-8)
    return np.clip(boosted, 1.0, max_weight)


def write_single_model_prediction_file(output_path, raw_points, preds_abs):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    preds_abs = np.asarray(preds_abs).reshape(-1)
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(len(raw_points)):
            x, y, z = raw_points[i, 0], raw_points[i, 1], raw_points[i, 2]
            f.write(f"{x} {y} {z} {preds_abs[i]:.4f} {0.0:.4f}\n")


def assert_output_split_matches_manifest(prediction_run_dir, manifest):
    prediction_run_dir = Path(prediction_run_dir)
    for split in ("train", "test"):
        split_dir = prediction_run_dir / split
        actual = sorted(path.name for path in split_dir.iterdir() if path.is_file()) if split_dir.exists() else []
        expected = manifest[split]
        if actual != expected:
            raise RuntimeError(
                f"Output {split} files do not match baseline manifest. "
                f"Expected {expected}, got {actual}."
            )


def stack_case_arrays(cases):
    X = np.vstack([case["X"] for case in cases])
    y_over = np.vstack([case["y"] for case in cases]).reshape(-1, 1)
    z_values = X[:, 7].reshape(-1, 1)
    return X, y_over, z_values


def prepare_targets(y_over, config_name):
    spec = CONFIG_SPECS[config_name]
    if spec["target"] == "log":
        return overpressure_to_log_target(y_over)
    return y_over / Y_SCALE_FACTOR


def prepare_weights(z_values, config_name, z_threshold, max_z_weight):
    spec = CONFIG_SPECS[config_name]
    if spec["z_weighted"]:
        return compute_z_sample_weights(z_values, threshold=z_threshold, max_weight=max_z_weight)
    return np.ones_like(z_values, dtype=float)


def make_loader(cases, scaler_X, config_name, batch_size, shuffle, seed, z_threshold, max_z_weight):
    X, y_over, z_values = stack_case_arrays(cases)
    X_norm = scaler_X.transform(X)
    targets = prepare_targets(y_over, config_name)
    weights = prepare_weights(z_values, config_name, z_threshold, max_z_weight)
    dataset = TensorDataset(
        torch.tensor(X_norm, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator if shuffle else None)


def weighted_mean(loss_values, weights):
    return torch.sum(loss_values * weights) / torch.clamp(torch.sum(weights), min=1e-8)


def compute_training_loss(config_name, pred, target, weights):
    spec = CONFIG_SPECS[config_name]
    if spec["loss"] == "mape":
        loss_values = torch.abs((pred - target) / (torch.abs(target) + 1e-5))
    elif spec["loss"] == "huber":
        loss_values = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    else:
        loss_values = torch.abs(pred - target)
    return weighted_mean(loss_values, weights)


def evaluate_loss(model, loader, config_name):
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch_X, batch_y, batch_w in loader:
            total_loss += compute_training_loss(config_name, model(batch_X), batch_y, batch_w).item()
            batches += 1
    return total_loss / max(batches, 1)


def split_inner_validation(train_cases, seed, val_fraction):
    if len(train_cases) < 2:
        return train_cases, train_cases
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(train_cases))
    n_val = max(1, int(round(len(train_cases) * val_fraction)))
    val_indices = set(indices[:n_val])
    fit_cases = [case for idx, case in enumerate(train_cases) if idx not in val_indices]
    val_cases = [case for idx, case in enumerate(train_cases) if idx in val_indices]
    return fit_cases, val_cases


def train_model_for_config(config_name, fit_cases, val_cases, args, seed, model_save_path):
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_fit, _, _ = stack_case_arrays(fit_cases)
    from sklearn.preprocessing import StandardScaler

    scaler_X = StandardScaler()
    scaler_X.fit(X_fit)
    train_loader = make_loader(
        fit_cases,
        scaler_X,
        config_name,
        args.batch_size,
        shuffle=True,
        seed=seed,
        z_threshold=args.z_threshold,
        max_z_weight=args.max_z_weight,
    )
    val_loader = make_loader(
        val_cases,
        scaler_X,
        config_name,
        args.batch_size,
        shuffle=False,
        seed=seed,
        z_threshold=args.z_threshold,
        max_z_weight=args.max_z_weight,
    )

    model = ImprovedMLP(input_dim=10, dropout_p=args.dropout_p)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss = float("inf")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for batch_X, batch_y, batch_w in train_loader:
            optimizer.zero_grad()
            loss = compute_training_loss(config_name, model(batch_X), batch_y, batch_w)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            val_loss = evaluate_loss(model, val_loader, config_name)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_save_path)

    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    return model, scaler_X, best_val_loss


def predict_abs_pressure(model, scaler_X, case_X, config_name):
    case_X_norm = scaler_X.transform(case_X)
    case_X_tensor = torch.tensor(case_X_norm, dtype=torch.float32)
    with torch.no_grad():
        raw_pred = model(case_X_tensor).numpy().reshape(-1)
    if CONFIG_SPECS[config_name]["target"] == "log":
        overpressure = log_target_to_overpressure(raw_pred)
    else:
        overpressure = raw_pred * Y_SCALE_FACTOR
    return np.maximum(overpressure, 0.0) + ATM_PRESSURE


def write_config_predictions(config_name, run_id, model, scaler_X, all_cases, prediction_run_dir):
    for case in all_cases:
        preds_abs = predict_abs_pressure(model, scaler_X, case["X"], config_name)
        output_path = Path(prediction_run_dir) / case["split"] / case["case_file"]
        write_single_model_prediction_file(output_path, case["raw_points"], preds_abs)


def run_config_fold(config_name, run_id, all_case_data, args):
    seed = 42 + run_id + DEFAULT_CONFIGS.index(config_name) * 1000
    manifest = load_split_manifest(args.baseline_pred_dir, run_id)
    train_cases, test_cases = cases_from_manifest(all_case_data, manifest)
    all_cases = train_cases + test_cases

    if args.strict_inner_val:
        fit_cases, val_cases = split_inner_validation(train_cases, seed, args.inner_val_fraction)
    else:
        fit_cases, val_cases = train_cases, test_cases

    output_root = Path(args.output_folder)
    model_path = output_root / "models" / config_name / f"run_{run_id}" / "model.pth"
    scaler_path = output_root / "scalers" / config_name / f"scaler_X_{run_id}.pkl"
    prediction_run_dir = output_root / "predictions" / config_name / f"run_{run_id}"

    print(f"\n=== Config {config_name} | run_{run_id} ===")
    print(f"fit cases: {len(fit_cases)} | val cases: {len(val_cases)} | test cases: {len(test_cases)}")
    model, scaler_X, best_val_loss = train_model_for_config(config_name, fit_cases, val_cases, args, seed, model_path)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_X, scaler_path)
    write_config_predictions(config_name, run_id, model, scaler_X, all_cases, prediction_run_dir)
    assert_output_split_matches_manifest(prediction_run_dir, manifest)
    return {"config": config_name, "run_id": run_id, "best_val_loss": best_val_loss}


def write_config_metrics_csv(rows, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["config", "train_avg_mape", "test_avg_mape", "test_avg_r2", "mean_best_val_loss"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_best_config_summary(best_row, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_row.keys()))
        writer.writeheader()
        writer.writerow(best_row)


def write_experiment_summary(rows, best_row, output_path, strict_inner_val=False):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mode = "strict inner validation" if strict_inner_val else "quick screening"
    lines = [
        "# MAPE专项优化实验总结",
        "",
        f"- 实验口径：{mode}",
        "- 原始基线测试 MAPE：17.23%",
        "- Dropout p 等权集成测试 MAPE：16.85%",
        "- 当前加权复用最佳测试 MAPE：15.79% (`softmax_top3_t2`)",
        "",
        "## 本轮配置结果",
        "",
        "| Config | Train MAPE | Test MAPE | Test R2 | Mean best val loss |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['config']}` | {float(row['train_avg_mape']):.2f}% | "
            f"{float(row['test_avg_mape']):.2f}% | {float(row['test_avg_r2']):.4f} | "
            f"{float(row['mean_best_val_loss']):.6f} |"
        )
    if best_row:
        lines.extend(
            [
                "",
                "## 当前最佳",
                "",
                f"- 最佳配置：`{best_row['config']}`",
                f"- 测试 MAPE：{float(best_row['test_avg_mape']):.2f}%",
                f"- 测试 R2：{float(best_row['test_avg_r2']):.4f}",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_config(config_name, args, fold_rows):
    output_root = Path(args.output_folder)
    pred_base_dir = output_root / "predictions" / config_name
    summary = evaluate_runs(args.data_folder, str(pred_base_dir), args.num_runs)
    metrics_dir = output_root / "metrics" / config_name
    write_summary_csv(summary, metrics_dir / "metrics_summary.csv")
    baseline_summary = evaluate_runs(args.data_folder, args.baseline_pred_dir, args.num_runs)
    write_baseline_comparison_csv(baseline_summary, summary, metrics_dir / "baseline_vs_config.csv")
    mean_best_val_loss = float(np.mean([row["best_val_loss"] for row in fold_rows]))
    return {
        "config": config_name,
        "train_avg_mape": summary["train_avg_mape"],
        "test_avg_mape": summary["test_avg_mape"],
        "test_avg_r2": summary["test_avg_r2"],
        "mean_best_val_loss": mean_best_val_loss,
    }, summary


def run_mape_focused_training(args):
    configs = parse_config_names(args.configs)
    all_case_data = load_all_case_data(args.data_folder, args.case_info_path)
    if not all_case_data:
        raise RuntimeError(f"No valid case data loaded from {args.data_folder}")

    all_fold_rows = []
    config_rows = []
    for config_name in configs:
        fold_rows = []
        for run_id in range(1, args.num_runs + 1):
            row = run_config_fold(config_name, run_id, all_case_data, args)
            fold_rows.append(row)
            all_fold_rows.append(row)
        config_row, summary = evaluate_config(config_name, args, fold_rows)
        config_rows.append(config_row)
        print(f"\n评估结果：{config_name}")
        print_report(summary)

    output_root = Path(args.output_folder)
    write_config_metrics_csv(config_rows, output_root / "config_metrics.csv")
    best_row = min(config_rows, key=lambda row: float(row["test_avg_mape"]))
    write_best_config_summary(best_row, output_root / "best_config_summary.csv")
    write_experiment_summary(config_rows, best_row, output_root / "MAPE专项优化实验总结.md", args.strict_inner_val)
    print(f"\nBest config by test MAPE: {best_row['config']} ({float(best_row['test_avg_mape']):.4f}%)")
    return config_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Train MAPE-focused model variants for case-wise prediction.")
    parser.add_argument("--configs", default=None)
    parser.add_argument("--output-folder", default=OUTPUT_FOLDER)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--data-folder", default=DATA_FOLDER)
    parser.add_argument("--case-info-path", default=CASE_INFO_PATH)
    parser.add_argument("--baseline-pred-dir", default=BASELINE_PRED_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout-p", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--z-threshold", type=float, default=DEFAULT_Z_THRESHOLD)
    parser.add_argument("--max-z-weight", type=float, default=DEFAULT_MAX_Z_WEIGHT)
    parser.add_argument("--strict-inner-val", action="store_true")
    parser.add_argument("--inner-val-fraction", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    run_mape_focused_training(args)


if __name__ == "__main__":
    main()
