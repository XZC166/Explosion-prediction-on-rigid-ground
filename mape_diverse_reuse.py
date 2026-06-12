import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch

from evaluate_metrics import evaluate_directory
from main_case_loop_ensemble import (
    BASELINE_PRED_DIR,
    DATA_FOLDER,
    ImprovedMLP,
    Y_SCALE_FACTOR,
    cases_from_manifest,
    load_all_case_data,
    load_split_manifest,
    make_loaders,
    predict_member,
    write_baseline_comparison_csv,
    write_prediction_file,
    write_summary_csv,
)


LEGACY_OUTPUT_FOLDER = "ensemble_outputs/mape_diverse_reuse"
OUTPUT_FOLDER = "ensemble_outputs/mape_diverse_reuse_strict"
PREVIOUS_STAGE_OUTPUT_FOLDER = "ensemble_outputs/weighted_reuse"
CURRENT_FAST_REFERENCE_MAPE = 15.26
PREVIOUS_STAGE_REFERENCE_MAPE = 15.79
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


@dataclass(frozen=True)
class MemberConfig:
    name: str
    dropout_p: float
    lr: float
    weight_decay: float
    seed_offset: int


DEFAULT_MEMBER_CONFIGS = [
    MemberConfig("p0_lr1e-3_wd0_s0", 0.0, 1e-3, 0.0, 0),
    MemberConfig("p01_lr1e-3_wd0_s1", 0.1, 1e-3, 0.0, 1),
    MemberConfig("p02_lr1e-3_wd1e-6_s0", 0.2, 1e-3, 1e-6, 0),
    MemberConfig("p03_lr1e-3_wd1e-6_s1", 0.3, 1e-3, 1e-6, 1),
    MemberConfig("p0_lr5e-4_wd1e-6_s1", 0.0, 5e-4, 1e-6, 1),
    MemberConfig("p01_lr5e-4_wd1e-5_s0", 0.1, 5e-4, 1e-5, 0),
    MemberConfig("p02_lr5e-4_wd0_s2", 0.2, 5e-4, 0.0, 2),
    MemberConfig("p03_lr5e-4_wd1e-5_s2", 0.3, 5e-4, 1e-5, 2),
]


def parse_member_configs(raw_configs):
    if raw_configs is None:
        return list(DEFAULT_MEMBER_CONFIGS)

    configs = []
    for raw_config in raw_configs.split(";"):
        raw_config = raw_config.strip()
        if not raw_config:
            continue
        parts = [part.strip() for part in raw_config.split(":")]
        if len(parts) != 5:
            raise ValueError(
                "Member config must use name:dropout_p:lr:weight_decay:seed_offset format."
            )
        name, dropout_p, lr, weight_decay, seed_offset = parts
        configs.append(
            MemberConfig(
                name=name,
                dropout_p=float(dropout_p),
                lr=float(lr),
                weight_decay=float(weight_decay),
                seed_offset=int(seed_offset),
            )
        )

    if not configs:
        raise ValueError("At least one member config is required.")
    validate_member_configs(configs)
    return configs


def validate_member_configs(configs):
    names = [config.name for config in configs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate member config name(s): {', '.join(duplicates)}")
    for config in configs:
        if config.dropout_p < 0.0 or config.dropout_p >= 1.0:
            raise ValueError(f"dropout_p must be in [0, 1): {config.name}")
        if config.lr <= 0.0:
            raise ValueError(f"lr must be positive: {config.name}")
        if config.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be non-negative: {config.name}")


def normalize_weights(weights):
    weights = np.asarray(weights, dtype=float)
    total = np.sum(weights)
    if total <= 0:
        raise ValueError("Strategy produced non-positive weights.")
    return weights / total


def member_validation_mape(row):
    if row.get("validation_mape_percent") not in (None, ""):
        return float(row["validation_mape_percent"])
    if row.get("best_val_mape_percent") not in (None, ""):
        return float(row["best_val_mape_percent"])
    raise KeyError("Member row is missing validation_mape_percent.")


def parse_strategy_names(raw_strategies):
    if raw_strategies is None:
        return list(DEFAULT_STRATEGIES)
    strategies = [strategy.strip() for strategy in raw_strategies.split(",") if strategy.strip()]
    if not strategies:
        raise ValueError("At least one strategy is required.")
    return strategies


def select_members_for_strategy(strategy, members):
    if not members:
        raise ValueError("No members available for strategy selection.")

    members = list(members)
    sorted_members = sorted(members, key=member_validation_mape)

    if strategy == "equal_all":
        selected = members
        weights = np.ones(len(selected), dtype=float)
    elif strategy.startswith("top"):
        k = int(strategy.replace("top", ""))
        selected = sorted_members[: min(k, len(sorted_members))]
        weights = np.ones(len(selected), dtype=float)
    elif strategy == "inv_mape_all":
        selected = members
        weights = [1.0 / max(member_validation_mape(row), 1e-8) for row in selected]
    elif strategy == "inv_mape_top3":
        selected = sorted_members[: min(3, len(sorted_members))]
        weights = [1.0 / max(member_validation_mape(row), 1e-8) for row in selected]
    elif strategy.startswith("softmax_top3_t"):
        temperature = float(strategy.replace("softmax_top3_t", ""))
        selected = sorted_members[: min(3, len(sorted_members))]
        raw = np.array(
            [-member_validation_mape(row) / temperature for row in selected],
            dtype=float,
        )
        raw = raw - np.max(raw)
        weights = np.exp(raw)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    if not selected:
        raise ValueError(f"Strategy selected no members: {strategy}")
    return selected, normalize_weights(weights)


def split_train_validation_cases(train_cases, validation_fraction, seed):
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1).")
    if len(train_cases) < 2:
        raise ValueError("At least two outer-train cases are required for validation split.")

    rng = np.random.default_rng(seed)
    shuffled_indices = np.arange(len(train_cases))
    rng.shuffle(shuffled_indices)
    validation_count = int(round(len(train_cases) * validation_fraction))
    validation_count = max(1, min(len(train_cases) - 1, validation_count))
    validation_indices = set(int(index) for index in shuffled_indices[:validation_count])

    inner_train_cases = []
    validation_cases = []
    for index, case in enumerate(train_cases):
        if index in validation_indices:
            validation_cases.append(dict(case, split="validation"))
        else:
            inner_train_cases.append(dict(case, split="train"))
    return inner_train_cases, validation_cases


def case_file_names(cases):
    return [case["case_file"] for case in cases]


def strict_manifest_from_cases(inner_train_cases, validation_cases, test_cases):
    return {
        "train": case_file_names(inner_train_cases),
        "validation": case_file_names(validation_cases),
        "test": case_file_names(test_cases),
    }


def write_fold_split_manifest(output_folder, run_id, manifest):
    output_path = Path(output_folder) / "manifests" / f"run_{run_id}_split_manifest.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "split", "case_file"])
        writer.writeheader()
        for split in ("train", "validation", "test"):
            for case_file in manifest.get(split, []):
                writer.writerow({"run_id": run_id, "split": split, "case_file": case_file})


def prepare_strict_fold_cases(run_id, all_case_data, args):
    outer_manifest = load_split_manifest(args.baseline_pred_dir, run_id)
    outer_train_cases, test_cases = cases_from_manifest(all_case_data, outer_manifest)
    inner_train_cases, validation_cases = split_train_validation_cases(
        outer_train_cases,
        args.validation_fraction,
        args.validation_seed + run_id,
    )
    manifest = strict_manifest_from_cases(inner_train_cases, validation_cases, test_cases)
    return inner_train_cases, validation_cases, test_cases, manifest


def train_diverse_member(train_loader, test_loader, config, model_save_path, epochs, batch_lr_seed, eval_every):
    seed = batch_lr_seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = ImprovedMLP(input_dim=10, dropout_p=config.dropout_p)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_val_loss = float("inf")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = torch.mean(torch.abs((pred - batch_y) / (batch_y + 1e-5)))
            loss.backward()
            optimizer.step()

        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            model.eval()
            val_loss = 0.0
            batches = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    val_loss += torch.mean(torch.abs((model(batch_X) - batch_y) / (batch_y + 1e-5))).item()
                    batches += 1
            val_loss /= max(batches, 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_save_path)

    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    return model, best_val_loss


def assert_prediction_split_matches_manifest(prediction_run_dir, manifest):
    prediction_run_dir = Path(prediction_run_dir)
    for split in ("train", "validation", "test"):
        split_dir = prediction_run_dir / split
        actual = sorted(path.name for path in split_dir.iterdir() if path.is_file()) if split_dir.exists() else []
        if actual != manifest.get(split, []):
            raise RuntimeError(
                f"Output {split} files do not match baseline manifest. "
                f"Expected {manifest.get(split, [])}, got {actual}."
            )


def fold_prediction_is_complete(output_folder, member_id, run_id, manifest):
    prediction_run_dir = Path(output_folder) / "member_predictions" / member_id / f"run_{run_id}"
    try:
        assert_prediction_split_matches_manifest(prediction_run_dir, manifest)
    except RuntimeError:
        return False
    return True


def write_member_predictions(config, run_id, model, scaler_X, all_cases, output_folder):
    prediction_run_dir = Path(output_folder) / "member_predictions" / config.name / f"run_{run_id}"
    for case in all_cases:
        preds_abs = predict_member(model, scaler_X, case["X"]).reshape(-1)
        preds_var = np.zeros_like(preds_abs)
        output_path = prediction_run_dir / case["split"] / case["case_file"]
        write_prediction_file(output_path, case["raw_points"], preds_abs, preds_var)
    return prediction_run_dir


def run_member_fold(run_id, config, all_case_data, args):
    seed = 42 + run_id + config.seed_offset * 1000
    train_cases, validation_cases, test_cases, manifest = prepare_strict_fold_cases(
        run_id,
        all_case_data,
        args,
    )
    all_cases = train_cases + validation_cases + test_cases

    scaler_X, train_loader, validation_loader, train_points, validation_points = make_loaders(
        train_cases,
        validation_cases,
        args.batch_size,
        seed,
    )
    test_points = sum(len(case["X"]) for case in test_cases)
    output_root = Path(args.output_folder)
    model_path = output_root / "models" / config.name / f"run_{run_id}" / "model.pth"
    scaler_path = output_root / "scalers" / config.name / f"scaler_X_{run_id}.pkl"
    write_fold_split_manifest(output_root, run_id, manifest)

    print(f"\n=== Member {config.name} | run_{run_id} ===")
    print(
        f"dropout={config.dropout_p:g} lr={config.lr:g} weight_decay={config.weight_decay:g} "
        f"seed={seed} train_points={train_points} validation_points={validation_points} "
        f"test_points={test_points}"
    )
    model, best_val_loss = train_diverse_member(
        train_loader,
        validation_loader,
        config,
        model_path,
        args.epochs,
        seed,
        args.eval_every,
    )
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_X, scaler_path)
    prediction_run_dir = write_member_predictions(config, run_id, model, scaler_X, all_cases, output_root)
    assert_prediction_split_matches_manifest(prediction_run_dir, manifest)
    print(f"best val MAPE: {best_val_loss * 100:.4f}%")
    return {
        "run_id": run_id,
        "member_id": config.name,
        "dropout_p": config.dropout_p,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "seed_offset": config.seed_offset,
        "validation_mape_percent": best_val_loss * 100.0,
    }


def read_member_metrics(member_metrics_path):
    grouped = {}
    for parsed in read_member_metric_rows(member_metrics_path):
        grouped.setdefault(parsed["run_id"], []).append(parsed)
    return grouped


def read_member_metric_rows(member_metrics_path):
    member_metrics_path = Path(member_metrics_path)
    if not member_metrics_path.exists():
        return []

    rows = []
    with open(member_metrics_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = dict(row)
            parsed["run_id"] = int(parsed["run_id"])
            for key in ("dropout_p", "lr", "weight_decay"):
                parsed[key] = float(parsed[key])
            if "validation_mape_percent" in parsed and parsed["validation_mape_percent"] != "":
                parsed["validation_mape_percent"] = float(parsed["validation_mape_percent"])
            elif "best_val_mape_percent" in parsed and parsed["best_val_mape_percent"] != "":
                parsed["validation_mape_percent"] = float(parsed["best_val_mape_percent"])
            parsed["seed_offset"] = int(parsed["seed_offset"])
            rows.append(parsed)
    return rows


def merge_member_metric_rows(existing_rows, new_rows):
    merged = {(row["member_id"], int(row["run_id"])): dict(row) for row in existing_rows}
    for row in new_rows:
        merged[(row["member_id"], int(row["run_id"]))] = dict(row)
    return [
        merged[key]
        for key in sorted(merged, key=lambda item: (item[0], item[1]))
    ]


def write_member_metrics_csv(rows, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "member_id",
        "dropout_p",
        "lr",
        "weight_decay",
        "seed_offset",
        "validation_mape_percent",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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


def write_strategy_predictions(strategy, output_folder, metrics_by_run, num_runs, splits=("train", "validation", "test")):
    output_folder = Path(output_folder)
    for run_id in range(1, num_runs + 1):
        selected, weights = select_members_for_strategy(strategy, metrics_by_run[run_id])
        selected_ids = [row["member_id"] for row in selected]
        reference_dir = output_folder / "member_predictions" / selected_ids[0] / f"run_{run_id}"
        for split in splits:
            split_dir = reference_dir / split
            for reference_file in sorted(path for path in split_dir.iterdir() if path.is_file()):
                arrays = [
                    load_prediction_array(
                        output_folder
                        / "member_predictions"
                        / member_id
                        / f"run_{run_id}"
                        / split
                        / reference_file.name
                    )
                    for member_id in selected_ids
                ]
                blended = blend_prediction_arrays(arrays, weights)
                write_prediction_array(
                    output_folder / "predictions" / strategy / f"run_{run_id}" / split / reference_file.name,
                    blended,
                )


def selected_members_by_run(strategy, metrics_by_run, num_runs):
    parts = []
    for run_id in range(1, num_runs + 1):
        selected, weights = select_members_for_strategy(strategy, metrics_by_run[run_id])
        formatted = ",".join(
            f"{row['member_id']}:w={weight:.4f}" for row, weight in zip(selected, weights)
        )
        parts.append(f"run_{run_id}[{formatted}]")
    return "; ".join(parts)


def write_strategy_metrics_csv(rows, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "strategy",
        "validation_avg_mape",
        "validation_avg_r2",
        "train_avg_mape",
        "test_avg_mape",
        "test_avg_r2",
        "selected_by_run",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_best_strategy_summary(best_row, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_row.keys()))
        writer.writeheader()
        writer.writerow(best_row)


def summarize_split_rows(rows, split_name):
    split_rows = [row for row in rows if row["split"] == split_name]
    return {
        "avg_mape": float(np.mean([row["mape_over"] for row in split_rows])) if split_rows else 0.0,
        "avg_r2": float(np.mean([row["r2"] for row in split_rows])) if split_rows else 0.0,
    }


def evaluate_prediction_splits(true_dir, pred_base_dir, num_runs, splits):
    rows = []
    for run_id in range(1, num_runs + 1):
        for split in splits:
            split_dir = Path(pred_base_dir) / f"run_{run_id}" / split
            result = evaluate_directory(true_dir, str(split_dir))
            if not result:
                continue
            rows.append({"run_id": run_id, "split": split, **result})

    summary = {
        "num_runs": num_runs,
        "rows": rows,
        "train_avg_mape": summarize_split_rows(rows, "train")["avg_mape"],
        "validation_avg_mape": summarize_split_rows(rows, "validation")["avg_mape"],
        "validation_avg_r2": summarize_split_rows(rows, "validation")["avg_r2"],
        "test_avg_mape": summarize_split_rows(rows, "test")["avg_mape"],
        "test_avg_r2": summarize_split_rows(rows, "test")["avg_r2"],
    }
    return summary


def print_split_report(summary):
    print(f"{'='*85}")
    print(
        f"{'Run':^5} | {'Split':^10} | {'Samples':^8} | "
        f"{'MAPE(%)':^12} | {'Abs MAPE(%)':^12} | {'R2 Score':^10} | {'RMSE (Pa)':^10}"
    )
    print(f"{'-'*85}")
    for row in summary["rows"]:
        print(
            f"{row['run_id']:^5} | {row['split']:^10} | {row['samples']:^8} | "
            f"{row['mape_over']:^12.2f} | {row['mape_abs']:^12.2f} | "
            f"{row['r2']:^10.4f} | {row['rmse']:^10.1f}"
        )
    print(f"{'-'*85}")
    print(
        f"Validation MAPE: {summary['validation_avg_mape']:.2f}% | "
        f"Test MAPE: {summary['test_avg_mape']:.2f}% | "
        f"Test R2: {summary['test_avg_r2']:.4f}"
    )
    print(f"{'='*85}")


def write_split_summary_csv(summary, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_id", "split", "samples", "mape_over", "mape_abs", "r2", "rmse"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow({key: row[key] for key in fieldnames})


def evaluate_strategy(strategy, args, splits):
    output_folder = Path(args.output_folder)
    pred_base_dir = output_folder / "predictions" / strategy
    summary = evaluate_prediction_splits(args.data_folder, pred_base_dir, args.num_runs, splits)
    metrics_dir = output_folder / "metrics" / strategy
    write_split_summary_csv(summary, metrics_dir / "metrics_summary.csv")
    return summary


def choose_best_strategy_by_validation(rows):
    if not rows:
        raise ValueError("No strategy rows available for validation selection.")
    return min(rows, key=lambda row: float(row["validation_avg_mape"]))


def read_best_strategy_summary(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def read_reference_result(source_dir, fallback_mape, fallback_strategy):
    summary = read_best_strategy_summary(Path(source_dir) / "best_strategy_summary.csv")
    if summary and summary.get("test_avg_mape") not in (None, ""):
        return {
            "strategy": summary.get("strategy", fallback_strategy),
            "test_mape": float(summary["test_avg_mape"]),
            "test_r2": summary.get("test_avg_r2", ""),
        }
    return {
        "strategy": fallback_strategy,
        "test_mape": fallback_mape,
        "test_r2": "",
    }


def write_comparison_summary(best_row, args):
    output_folder = Path(args.output_folder)
    current_reference = read_reference_result(
        LEGACY_OUTPUT_FOLDER,
        args.current_fast_reference_mape,
        "softmax_top3_t2",
    )
    previous_reference = read_reference_result(
        PREVIOUS_STAGE_OUTPUT_FOLDER,
        args.previous_stage_reference_mape,
        "softmax_top3_t2",
    )
    rows = [
        {
            "result": "strict_validation_selected",
            "strategy": best_row["strategy"],
            "selection_metric": "validation_avg_mape",
            "validation_mape": best_row.get("validation_avg_mape", ""),
            "test_mape": best_row.get("test_avg_mape", ""),
            "test_r2": best_row.get("test_avg_r2", ""),
            "delta_vs_current_15_26": (
                float(best_row["test_avg_mape"]) - current_reference["test_mape"]
                if best_row.get("test_avg_mape") not in (None, "")
                else ""
            ),
            "delta_vs_previous_15_79": (
                float(best_row["test_avg_mape"]) - previous_reference["test_mape"]
                if best_row.get("test_avg_mape") not in (None, "")
                else ""
            ),
            "source": str(output_folder),
        },
        {
            "result": "current_fast_reference",
            "strategy": current_reference["strategy"],
            "selection_metric": "outer_test_fast_trial",
            "validation_mape": "",
            "test_mape": current_reference["test_mape"],
            "test_r2": current_reference["test_r2"],
            "delta_vs_current_15_26": 0.0,
            "delta_vs_previous_15_79": current_reference["test_mape"] - previous_reference["test_mape"],
            "source": LEGACY_OUTPUT_FOLDER,
        },
        {
            "result": "previous_stage_reference",
            "strategy": previous_reference["strategy"],
            "selection_metric": "outer_test_fast_trial",
            "validation_mape": "",
            "test_mape": previous_reference["test_mape"],
            "test_r2": previous_reference["test_r2"],
            "delta_vs_current_15_26": previous_reference["test_mape"] - current_reference["test_mape"],
            "delta_vs_previous_15_79": 0.0,
            "source": PREVIOUS_STAGE_OUTPUT_FOLDER,
        },
    ]
    output_path = output_folder / "comparison_summary.csv"
    fieldnames = [
        "result",
        "strategy",
        "selection_metric",
        "validation_mape",
        "test_mape",
        "test_r2",
        "delta_vs_current_15_26",
        "delta_vs_previous_15_79",
        "source",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def member_config_repro_arg(raw_member_configs):
    if raw_member_configs:
        return f' --member-configs "{raw_member_configs}"'
    return ""


def strategy_repro_arg(raw_strategies):
    if raw_strategies:
        return f' --strategies "{raw_strategies}"'
    return ""


def write_repro_commands(args):
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    script_name = Path(__file__).name
    train_command = (
        f"python {script_name}"
        f" --output-folder {args.output_folder}"
        f" --num-runs {args.num_runs}"
        f" --epochs {args.epochs}"
        f" --batch-size {args.batch_size}"
        f" --eval-every {args.eval_every}"
        f" --validation-fraction {args.validation_fraction}"
        f" --validation-seed {args.validation_seed}"
        f" --data-folder {args.data_folder}"
        f" --case-info-path {args.case_info_path}"
        f" --baseline-pred-dir {args.baseline_pred_dir}"
        f"{member_config_repro_arg(args.member_configs)}"
        f"{strategy_repro_arg(args.strategies)}"
    )
    reuse_command = train_command + " --reuse-only"
    test_command = "python -m unittest discover -s tests"
    with open(output_folder / "repro_commands.txt", "w", encoding="utf-8") as f:
        f.write("Strict train/validation/test review:\n")
        f.write(train_command + "\n\n")
        f.write("Reuse existing strict members and rerun validation strategy selection:\n")
        f.write(reuse_command + "\n\n")
        f.write("Unit tests:\n")
        f.write(test_command + "\n\n")
        f.write("Original argv:\n")
        f.write(" ".join(sys.argv) + "\n")


def run_strategy_reuse(args):
    output_folder = Path(args.output_folder)
    metrics_by_run = read_member_metrics(output_folder / "member_metrics.csv")
    strategies = parse_strategy_names(args.strategies)
    rows = []
    for strategy in strategies:
        print(f"\n=== Validation strategy: {strategy} ===")
        write_strategy_predictions(
            strategy,
            output_folder,
            metrics_by_run,
            args.num_runs,
            splits=("validation",),
        )
        summary = evaluate_strategy(strategy, args, splits=("validation",))
        print_split_report(summary)
        rows.append(
            {
                "strategy": strategy,
                "validation_avg_mape": summary["validation_avg_mape"],
                "validation_avg_r2": summary["validation_avg_r2"],
                "selected_by_run": selected_members_by_run(strategy, metrics_by_run, args.num_runs),
            }
        )
    write_strategy_metrics_csv(rows, output_folder / "strategy_metrics.csv")

    best_row = choose_best_strategy_by_validation(rows)
    print(
        f"\nBest strategy by validation MAPE: {best_row['strategy']} "
        f"({float(best_row['validation_avg_mape']):.4f}%)"
    )
    print("\n=== Final outer-test evaluation for validation-selected strategy ===")
    write_strategy_predictions(
        best_row["strategy"],
        output_folder,
        metrics_by_run,
        args.num_runs,
        splits=("train", "test"),
    )
    final_summary = evaluate_strategy(best_row["strategy"], args, splits=("train", "validation", "test"))
    print_split_report(final_summary)
    best_row = {
        **best_row,
        "train_avg_mape": final_summary["train_avg_mape"],
        "test_avg_mape": final_summary["test_avg_mape"],
        "test_avg_r2": final_summary["test_avg_r2"],
    }
    write_best_strategy_summary(best_row, output_folder / "best_strategy_summary.csv")
    write_comparison_summary(best_row, args)
    print(
        f"\nFinal test MAPE for validation-selected strategy {best_row['strategy']}: "
        f"{float(best_row['test_avg_mape']):.4f}%"
    )
    return rows


def run_training(args):
    configs = parse_member_configs(args.member_configs)
    all_case_data = load_all_case_data(args.data_folder, args.case_info_path)
    if not all_case_data:
        raise RuntimeError(f"No valid case data loaded from {args.data_folder}")

    output_root = Path(args.output_folder)
    metrics_path = output_root / "member_metrics.csv"
    rows = read_member_metric_rows(metrics_path)
    completed_keys = {(row["member_id"], int(row["run_id"])) for row in rows}
    for config in configs:
        for run_id in range(1, args.num_runs + 1):
            _, _, _, manifest = prepare_strict_fold_cases(run_id, all_case_data, args)
            key = (config.name, run_id)
            if key in completed_keys and fold_prediction_is_complete(output_root, config.name, run_id, manifest):
                print(f"Skipping completed member {config.name} | run_{run_id}")
                continue

            row = run_member_fold(run_id, config, all_case_data, args)
            rows = merge_member_metric_rows(rows, [row])
            completed_keys.add(key)
            write_member_metrics_csv(rows, metrics_path)
    return rows


def run_experiment(args):
    write_repro_commands(args)
    if not args.reuse_only:
        run_training(args)
    if not args.train_only:
        return run_strategy_reuse(args)
    return []


def parse_args():
    parser = argparse.ArgumentParser(description="Train diverse MAPE-loss members and reuse weighted inference.")
    parser.add_argument("--member-configs", default=None)
    parser.add_argument("--strategies", default=None)
    parser.add_argument("--output-folder", default=OUTPUT_FOLDER)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--validation-seed", type=int, default=20260609)
    parser.add_argument("--data-folder", default=DATA_FOLDER)
    parser.add_argument("--case-info-path", default="data/case_info.csv")
    parser.add_argument("--baseline-pred-dir", default=BASELINE_PRED_DIR)
    parser.add_argument("--current-fast-reference-mape", type=float, default=CURRENT_FAST_REFERENCE_MAPE)
    parser.add_argument("--previous-stage-reference-mape", type=float, default=PREVIOUS_STAGE_REFERENCE_MAPE)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--reuse-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
