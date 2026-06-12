import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn

from main_case_loop_ensemble import (
    ATM_PRESSURE,
    BATCH_SIZE,
    DATA_FOLDER,
    ImprovedMLP,
    Y_SCALE_FACTOR,
    load_all_case_data,
    make_loaders,
    predict_member,
    write_prediction_file,
)
from mape_diverse_reuse import (
    DEFAULT_MEMBER_CONFIGS,
    MemberConfig,
    assert_prediction_split_matches_manifest,
    blend_prediction_arrays,
    evaluate_prediction_splits,
    fold_prediction_is_complete,
    load_prediction_array,
    merge_member_metric_rows,
    parse_member_configs,
    read_member_metric_rows,
    train_diverse_member,
    write_member_metrics_csv,
    write_member_predictions,
    write_prediction_array,
    write_split_summary_csv,
)


OUTPUT_FOLDER = "ensemble_outputs/mape_single_fold_screening"
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


class SignedOutputMLP(nn.Module):
    def __init__(self, input_dim, dropout_p):
        super(SignedOutputMLP, self).__init__()
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
        )

    def forward(self, x):
        return self.net(x)


def make_screening_model(dropout_p, allow_negative_output):
    if allow_negative_output:
        return SignedOutputMLP(input_dim=10, dropout_p=dropout_p)
    return ImprovedMLP(input_dim=10, dropout_p=dropout_p)


def train_screening_member(
    train_loader,
    validation_loader,
    config,
    model_save_path,
    epochs,
    seed,
    eval_every,
    allow_negative_output,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = make_screening_model(config.dropout_p, allow_negative_output)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_val_loss = float("inf")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = torch.mean(torch.abs((pred - batch_y) / (torch.abs(batch_y) + 1e-5)))
            loss.backward()
            optimizer.step()

        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            model.eval()
            val_loss = 0.0
            batches = 0
            with torch.no_grad():
                for batch_X, batch_y in validation_loader:
                    val_loss += torch.mean(
                        torch.abs((model(batch_X) - batch_y) / (torch.abs(batch_y) + 1e-5))
                    ).item()
                    batches += 1
            val_loss /= max(batches, 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_save_path)

    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    return model, best_val_loss


def generate_case_split(all_cases, test_fraction, validation_fraction, seed):
    if len(all_cases) < 3:
        raise ValueError("At least three cases are required for train/validation/test split.")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1).")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1).")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(all_cases))
    rng.shuffle(indices)

    test_count = int(round(len(all_cases) * test_fraction))
    test_count = max(1, min(len(all_cases) - 2, test_count))
    remaining = indices[test_count:]
    validation_count = int(round(len(remaining) * validation_fraction))
    validation_count = max(1, min(len(remaining) - 1, validation_count))

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


def case_difficulty_bucket(case):
    features = np.asarray(case["X"], dtype=float)
    if features.ndim != 2 or features.shape[1] < 8:
        raise ValueError("Case features must be a 2D array with at least 8 columns.")

    blast = float(np.median(features[:, 3]))
    z_scaled = features[:, 7]
    median_z = float(np.median(z_scaled))
    max_z = float(np.max(z_scaled))

    if blast <= 1.0:
        return "low_charge_far" if median_z >= 2500.0 or max_z >= 6000.0 else "low_charge_near"
    if blast < 100.0:
        return "mid_charge_far" if median_z >= 1000.0 else "mid_charge_near"
    return "high_charge_far" if median_z >= 250.0 else "high_charge_near"


def _split_bucket_cases(bucket_cases, test_fraction, validation_fraction, rng):
    indices = np.arange(len(bucket_cases))
    rng.shuffle(indices)
    if len(bucket_cases) < 3:
        return list(indices[1:]), list(indices[:1]), []

    test_count = int(round(len(bucket_cases) * test_fraction))
    test_count = max(1, min(len(bucket_cases) - 2, test_count))
    remaining = indices[test_count:]
    validation_count = int(round(len(remaining) * validation_fraction))
    validation_count = max(1, min(len(remaining) - 1, validation_count))
    return list(remaining[validation_count:]), list(remaining[:validation_count]), list(indices[:test_count])


def generate_stratified_case_split(all_cases, test_fraction, validation_fraction, seed):
    if len(all_cases) < 3:
        raise ValueError("At least three cases are required for train/validation/test split.")
    buckets = {}
    for case in all_cases:
        buckets.setdefault(case_difficulty_bucket(case), []).append(case)

    rng = np.random.default_rng(seed)
    train_cases = []
    validation_cases = []
    test_cases = []
    for bucket_name in sorted(buckets):
        bucket_cases = buckets[bucket_name]
        train_indices, validation_indices, test_indices = _split_bucket_cases(
            bucket_cases,
            test_fraction,
            validation_fraction,
            rng,
        )
        train_cases.extend(dict(bucket_cases[index], split="train") for index in train_indices)
        validation_cases.extend(dict(bucket_cases[index], split="validation") for index in validation_indices)
        test_cases.extend(dict(bucket_cases[index], split="test") for index in test_indices)

    train_cases = sorted(train_cases, key=lambda case: case["case_file"])
    validation_cases = sorted(validation_cases, key=lambda case: case["case_file"])
    test_cases = sorted(test_cases, key=lambda case: case["case_file"])
    return CaseSplit(train_cases, validation_cases, test_cases)


def split_manifest(case_split):
    return {
        "train": [case["case_file"] for case in case_split.train_cases],
        "validation": [case["case_file"] for case in case_split.validation_cases],
        "test": [case["case_file"] for case in case_split.test_cases],
    }


def write_screening_manifest(manifest, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "split", "case_file"])
        writer.writeheader()
        for split in ("train", "validation", "test"):
            for case_file in manifest[split]:
                writer.writerow({"run_id": 1, "split": split, "case_file": case_file})


def parse_strategy_names(raw_strategies):
    if raw_strategies is None:
        return list(DEFAULT_SCREENING_STRATEGIES)
    strategies = [strategy.strip() for strategy in raw_strategies.split(",") if strategy.strip()]
    if not strategies:
        raise ValueError("At least one strategy is required.")
    return strategies


def member_validation_mape(row):
    value = row.get("validation_mape_percent", row.get("best_val_mape_percent"))
    if value in (None, ""):
        raise KeyError("Member row is missing validation_mape_percent.")
    return float(value)


def normalize_weights(weights):
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("Strategy produced non-positive weights.")
    return weights / total


def select_rule_based_members(strategy, members):
    sorted_members = sorted(members, key=member_validation_mape)
    if strategy == "top2":
        selected = sorted_members[: min(2, len(sorted_members))]
        weights = np.ones(len(selected), dtype=float)
    elif strategy in ("top3", "equal_top3"):
        selected = sorted_members[: min(3, len(sorted_members))]
        weights = np.ones(len(selected), dtype=float)
    elif strategy == "inv_mape_top3":
        selected = sorted_members[: min(3, len(sorted_members))]
        weights = [1.0 / max(member_validation_mape(row), 1e-8) for row in selected]
    elif strategy.startswith("softmax_top3_t"):
        temperature = float(strategy.replace("softmax_top3_t", ""))
        selected = sorted_members[: min(3, len(sorted_members))]
        raw = np.array([-member_validation_mape(row) / temperature for row in selected], dtype=float)
        raw = raw - np.max(raw)
        weights = np.exp(raw)
    else:
        raise ValueError(f"Unsupported rule-based strategy: {strategy}")
    if not selected:
        raise ValueError(f"Strategy selected no members: {strategy}")
    return selected, normalize_weights(weights)


def collect_validation_arrays(output_folder, selected_members, true_dir):
    output_folder = Path(output_folder)
    first_member = selected_members[0]["member_id"]
    validation_dir = output_folder / "member_predictions" / first_member / "run_1" / "validation"
    true_values = []
    member_values = [[] for _ in selected_members]

    for reference_file in sorted(path for path in validation_dir.iterdir() if path.is_file()):
        true_path = Path(true_dir) / reference_file.name
        true_data = np.loadtxt(true_path)
        if true_data.ndim == 1:
            true_data = true_data.reshape(1, -1)
        true_values.append(true_data[:, 3])
        for index, member in enumerate(selected_members):
            pred_path = (
                output_folder
                / "member_predictions"
                / member["member_id"]
                / "run_1"
                / "validation"
                / reference_file.name
            )
            pred_data = load_prediction_array(pred_path)
            member_values[index].append(pred_data[:, 3])

    if not true_values:
        raise ValueError("No validation predictions found for weight optimization.")

    y_true_abs = np.concatenate(true_values)
    member_pred_abs = np.stack([np.concatenate(values) for values in member_values], axis=0)
    return y_true_abs, member_pred_abs


def mape_for_weights(weights, y_true_abs, member_pred_abs):
    pred_abs = np.average(member_pred_abs, axis=0, weights=weights)
    y_true_over = y_true_abs - ATM_PRESSURE
    pred_over = pred_abs - ATM_PRESSURE
    return float(np.mean(np.abs((y_true_over - pred_over) / (np.abs(y_true_over) + 1e-5))) * 100.0)


def optimize_validation_weights(selected_members, output_folder, true_dir, l2_penalty):
    y_true_abs, member_pred_abs = collect_validation_arrays(output_folder, selected_members, true_dir)
    n_members = len(selected_members)
    equal_weights = np.ones(n_members, dtype=float) / n_members

    try:
        from scipy.optimize import minimize
    except Exception:
        return equal_weights

    def objective(weights):
        weights = normalize_weights(np.maximum(weights, 0.0))
        penalty = l2_penalty * float(np.sum((weights - equal_weights) ** 2))
        return mape_for_weights(weights, y_true_abs, member_pred_abs) + penalty

    result = minimize(
        objective,
        equal_weights,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_members,
        constraints=({"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},),
        options={"maxiter": 200, "ftol": 1e-9},
    )
    if not result.success:
        return equal_weights
    return normalize_weights(np.maximum(result.x, 0.0))


def select_members_for_strategy(strategy, members, output_folder=None, true_dir=None, weight_opt_top_k=4, weight_opt_l2=0.01):
    if strategy == "validation_weight_opt":
        sorted_members = sorted(members, key=member_validation_mape)
        selected = sorted_members[: min(weight_opt_top_k, len(sorted_members))]
        if output_folder is None or true_dir is None:
            weights = np.ones(len(selected), dtype=float)
        else:
            weights = optimize_validation_weights(selected, output_folder, true_dir, weight_opt_l2)
        return selected, normalize_weights(weights)
    return select_rule_based_members(strategy, members)


def select_best_strategy_by_validation(rows):
    if not rows:
        raise ValueError("No strategy rows available for validation selection.")
    return min(rows, key=lambda row: float(row["validation_avg_mape"]))


def selected_members_text(selected, weights):
    return ",".join(f"{row['member_id']}:w={weight:.6f}" for row, weight in zip(selected, weights))


def write_baseline_predictions(model, scaler_X, all_cases, output_folder):
    prediction_run_dir = Path(output_folder) / "baseline_predictions" / "run_1"
    for case in all_cases:
        preds_abs = predict_member(model, scaler_X, case["X"]).reshape(-1)
        preds_var = np.zeros_like(preds_abs)
        output_path = prediction_run_dir / case["split"] / case["case_file"]
        write_prediction_file(output_path, case["raw_points"], preds_abs, preds_var)
    return prediction_run_dir


def train_baseline(case_split, manifest, args):
    seed = args.random_seed
    scaler_X, train_loader, validation_loader, train_points, validation_points = make_loaders(
        case_split.train_cases,
        case_split.validation_cases,
        args.batch_size,
        seed,
    )
    model_path = Path(args.output_folder) / "baseline_model" / "model.pth"
    scaler_path = Path(args.output_folder) / "baseline_model" / "scaler_X.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_config = MemberConfig("baseline", args.baseline_dropout, args.baseline_lr, args.baseline_weight_decay, 0)
    print(
        f"\n=== Baseline | train_points={train_points} validation_points={validation_points} "
        f"dropout={args.baseline_dropout:g} lr={args.baseline_lr:g} ==="
    )
    model, best_val_loss = train_screening_member(
        train_loader,
        validation_loader,
        baseline_config,
        model_path,
        args.epochs,
        seed,
        args.eval_every,
        args.allow_negative_output,
    )
    joblib.dump(scaler_X, scaler_path)
    all_cases = case_split.train_cases + case_split.validation_cases + case_split.test_cases
    prediction_run_dir = write_baseline_predictions(model, scaler_X, all_cases, args.output_folder)
    assert_prediction_split_matches_manifest(prediction_run_dir, manifest)
    print(f"Baseline best validation MAPE: {best_val_loss * 100:.4f}%")
    return best_val_loss * 100.0


def train_member(config, case_split, manifest, args):
    seed = args.random_seed + config.seed_offset * 1000
    scaler_X, train_loader, validation_loader, train_points, validation_points = make_loaders(
        case_split.train_cases,
        case_split.validation_cases,
        args.batch_size,
        seed,
    )
    output_root = Path(args.output_folder)
    model_path = output_root / "models" / config.name / "run_1" / "model.pth"
    scaler_path = output_root / "scalers" / config.name / "scaler_X_1.pkl"

    print(
        f"\n=== Member {config.name} | train_points={train_points} validation_points={validation_points} ==="
    )
    print(
        f"dropout={config.dropout_p:g} lr={config.lr:g} "
        f"weight_decay={config.weight_decay:g} seed={seed}"
    )
    model, best_val_loss = train_screening_member(
        train_loader,
        validation_loader,
        config,
        model_path,
        args.epochs,
        seed,
        args.eval_every,
        args.allow_negative_output,
    )
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_X, scaler_path)
    all_cases = case_split.train_cases + case_split.validation_cases + case_split.test_cases
    prediction_run_dir = write_member_predictions(config, 1, model, scaler_X, all_cases, args.output_folder)
    assert_prediction_split_matches_manifest(prediction_run_dir, manifest)
    print(f"Member best validation MAPE: {best_val_loss * 100:.4f}%")
    return {
        "run_id": 1,
        "member_id": config.name,
        "dropout_p": config.dropout_p,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "seed_offset": config.seed_offset,
        "validation_mape_percent": best_val_loss * 100.0,
    }


def write_strategy_predictions(strategy, output_folder, members, weights, splits):
    output_folder = Path(output_folder)
    selected_ids = [row["member_id"] for row in members]
    reference_dir = output_folder / "member_predictions" / selected_ids[0] / "run_1"
    for split in splits:
        split_dir = reference_dir / split
        for reference_file in sorted(path for path in split_dir.iterdir() if path.is_file()):
            arrays = [
                load_prediction_array(
                    output_folder / "member_predictions" / member_id / "run_1" / split / reference_file.name
                )
                for member_id in selected_ids
            ]
            blended = blend_prediction_arrays(arrays, weights)
            write_prediction_array(
                output_folder / "predictions" / strategy / "run_1" / split / reference_file.name,
                blended,
            )


def evaluate_split_base(args, pred_base_dir, splits):
    return evaluate_prediction_splits(args.data_folder, pred_base_dir, 1, splits)


def write_strategy_metrics(rows, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "strategy",
        "validation_avg_mape",
        "validation_avg_r2",
        "train_avg_mape",
        "test_avg_mape",
        "test_avg_r2",
        "selected_members",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_best_strategy(best_row, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(best_row.keys()))
        writer.writeheader()
        writer.writerow(best_row)


def write_candidate_vs_baseline(best_row, baseline_summary, output_path):
    baseline_test = baseline_summary["test_avg_mape"]
    candidate_test = float(best_row["test_avg_mape"])
    row = {
        "baseline_test_mape": baseline_test,
        "candidate_strategy": best_row["strategy"],
        "candidate_validation_mape": best_row["validation_avg_mape"],
        "candidate_test_mape": candidate_test,
        "candidate_test_r2": best_row["test_avg_r2"],
        "delta_mape_candidate_minus_baseline": candidate_test - baseline_test,
        "below_15_percent": candidate_test < 15.0,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_repro_commands(args):
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    script_name = Path(__file__).name
    command = (
        f"python {script_name}"
        f" --output-folder {args.output_folder}"
        f" --epochs {args.epochs}"
        f" --eval-every {args.eval_every}"
        f" --batch-size {args.batch_size}"
        f" --random-seed {args.random_seed}"
        f" --test-fraction {args.test_fraction}"
        f" --validation-fraction {args.validation_fraction}"
        f" --data-folder {args.data_folder}"
        f" --case-info-path {args.case_info_path}"
    )
    if args.member_configs:
        command += f' --member-configs "{args.member_configs}"'
    if args.strategies:
        command += f' --strategies "{args.strategies}"'
    with open(output_folder / "repro_commands.txt", "w", encoding="utf-8") as f:
        f.write("Single-fold strict screening:\n")
        f.write(command + "\n\n")
        f.write("Reuse trained members and rerun validation strategy selection:\n")
        f.write(command + " --reuse-only\n")


def prepare_case_split(args):
    all_case_data = load_all_case_data(args.data_folder, args.case_info_path)
    if not all_case_data:
        raise RuntimeError(f"No valid case data loaded from {args.data_folder}")
    if args.split_mode == "random":
        case_split = generate_case_split(
            all_case_data,
            test_fraction=args.test_fraction,
            validation_fraction=args.validation_fraction,
            seed=args.random_seed,
        )
    elif args.split_mode == "stratified":
        case_split = generate_stratified_case_split(
            all_case_data,
            test_fraction=args.test_fraction,
            validation_fraction=args.validation_fraction,
            seed=args.random_seed,
        )
    else:
        raise ValueError(f"Unsupported split mode: {args.split_mode}")
    manifest = split_manifest(case_split)
    write_screening_manifest(manifest, Path(args.output_folder) / "split_manifest.csv")
    return case_split, manifest


def run_training(args):
    configs = parse_member_configs(args.member_configs)
    case_split, manifest = prepare_case_split(args)

    baseline_dir = Path(args.output_folder) / "baseline_predictions" / "run_1"
    if not args.skip_baseline and not (args.reuse_only and baseline_dir.exists()):
        train_baseline(case_split, manifest, args)

    metrics_path = Path(args.output_folder) / "member_metrics.csv"
    rows = read_member_metric_rows(metrics_path)
    completed = {(row["member_id"], int(row["run_id"])) for row in rows}
    for config in configs:
        key = (config.name, 1)
        if key in completed and fold_prediction_is_complete(args.output_folder, config.name, 1, manifest):
            print(f"Skipping completed member {config.name}")
            continue
        if args.reuse_only:
            raise RuntimeError(f"Missing completed predictions for member {config.name}; cannot reuse only.")
        row = train_member(config, case_split, manifest, args)
        rows = merge_member_metric_rows(rows, [row])
        completed.add(key)
        write_member_metrics_csv(rows, metrics_path)
    return case_split, manifest


def run_strategy_selection(args):
    output_folder = Path(args.output_folder)
    members = read_member_metric_rows(output_folder / "member_metrics.csv")
    if not members:
        raise RuntimeError("No member metrics found. Train members before strategy selection.")

    strategies = parse_strategy_names(args.strategies)
    rows = []
    selected_cache = {}
    for strategy in strategies:
        print(f"\n=== Validation strategy: {strategy} ===")
        selected, weights = select_members_for_strategy(
            strategy,
            members,
            output_folder=args.output_folder,
            true_dir=args.data_folder,
            weight_opt_top_k=args.weight_opt_top_k,
            weight_opt_l2=args.weight_opt_l2,
        )
        selected_cache[strategy] = (selected, weights)
        write_strategy_predictions(strategy, output_folder, selected, weights, splits=("validation",))
        summary = evaluate_split_base(args, output_folder / "predictions" / strategy, splits=("validation",))
        row = {
            "strategy": strategy,
            "validation_avg_mape": summary["validation_avg_mape"],
            "validation_avg_r2": summary["validation_avg_r2"],
            "selected_members": selected_members_text(selected, weights),
        }
        rows.append(row)
        print(
            f"Validation MAPE={summary['validation_avg_mape']:.4f}% "
            f"members={row['selected_members']}"
        )

    best_row = select_best_strategy_by_validation(rows)
    best_strategy = best_row["strategy"]
    selected, weights = selected_cache[best_strategy]
    print(f"\nBest strategy by validation MAPE: {best_strategy}")
    write_strategy_predictions(best_strategy, output_folder, selected, weights, splits=("train", "validation", "test"))
    final_summary = evaluate_split_base(
        args,
        output_folder / "predictions" / best_strategy,
        splits=("train", "validation", "test"),
    )
    best_row = {
        **best_row,
        "train_avg_mape": final_summary["train_avg_mape"],
        "test_avg_mape": final_summary["test_avg_mape"],
        "test_avg_r2": final_summary["test_avg_r2"],
    }
    write_strategy_metrics(rows, output_folder / "strategy_metrics.csv")
    write_best_strategy(best_row, output_folder / "best_strategy_summary.csv")
    write_split_summary_csv(final_summary, output_folder / "metrics" / best_strategy / "metrics_summary.csv")

    baseline_dir = output_folder / "baseline_predictions"
    if baseline_dir.exists():
        baseline_summary = evaluate_split_base(args, baseline_dir, splits=("train", "validation", "test"))
        write_split_summary_csv(baseline_summary, output_folder / "baseline_summary.csv")
        write_candidate_vs_baseline(best_row, baseline_summary, output_folder / "candidate_vs_baseline.csv")
        print(
            f"Baseline Test MAPE={baseline_summary['test_avg_mape']:.4f}% | "
            f"Candidate Test MAPE={float(best_row['test_avg_mape']):.4f}% | "
            f"Delta={float(best_row['test_avg_mape']) - baseline_summary['test_avg_mape']:.4f}%"
        )
    else:
        print(f"Candidate Test MAPE={float(best_row['test_avg_mape']):.4f}%")
    return best_row


def run_experiment(args):
    write_repro_commands(args)
    run_training(args)
    if args.train_only:
        return None
    return run_strategy_selection(args)


def parse_args():
    parser = argparse.ArgumentParser(description="Strict single-fold MAPE screening.")
    parser.add_argument("--output-folder", default=OUTPUT_FOLDER)
    parser.add_argument("--data-folder", default=DATA_FOLDER)
    parser.add_argument("--case-info-path", default="data/case_info.csv")
    parser.add_argument("--member-configs", default=None)
    parser.add_argument("--strategies", default=None)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--random-seed", type=int, default=20260611)
    parser.add_argument("--split-mode", choices=["random", "stratified"], default="stratified")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--baseline-dropout", type=float, default=0.0)
    parser.add_argument("--baseline-lr", type=float, default=1e-3)
    parser.add_argument("--baseline-weight-decay", type=float, default=0.0)
    parser.add_argument("--weight-opt-top-k", type=int, default=4)
    parser.add_argument("--weight-opt-l2", type=float, default=0.01)
    parser.add_argument("--reuse-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--allow-negative-output", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
