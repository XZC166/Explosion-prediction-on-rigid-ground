import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from main_case_loop_ensemble import (
    ATM_PRESSURE,
    BATCH_SIZE,
    DATA_FOLDER,
    EPOCHS,
    ImprovedMLP,
    LR,
    Y_SCALE_FACTOR,
    cases_from_manifest,
    load_split_manifest,
    make_loaders,
    mape_loss,
    predict_member,
    write_prediction_file,
)
from mape_single_fold_screening import (
    CaseSplit,
    case_difficulty_bucket,
    generate_stratified_case_split,
    load_all_case_data,
    split_manifest,
    write_screening_manifest,
)


OUTPUT_FOLDER = "ensemble_outputs/mape_rf_screening"
FORMAL_OUTPUT_FOLDER = "ensemble_outputs/mape_rf_screening_formal_stratified_fivefold"
FORMAL_RF_CONFIG = "rf200_l2_d16_clip5e4:200:2:16:2002:50000"
DEFAULT_FORMAL_CANDIDATE_CONFIG = "rf200_l2_d16_clip5e4"
DEFAULT_RF_CONFIGS = [
    "rf100_l3_d16_clip5e4:100:3:16:1003:50000",
    "rf150_l3_d16_clip5e4:150:3:16:1503:50000",
    "rf200_l2_d16_clip5e4:200:2:16:2002:50000",
    "rf200_l1_d16_clip5e4:200:1:16:2001:50000",
    "rf150_l2_none_clip5e4:150:2:none:1502:50000",
]
DEFAULT_STRATIFIED_SEEDS = [20260611, 20260612, 20260613, 20260614, 20260615]


@dataclass(frozen=True)
class RFConfig:
    name: str
    n_estimators: int
    min_samples_leaf: int
    max_depth: int | None
    random_state: int
    clip_denominator: float


def signed_log_target(values):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def inverse_signed_log_target(values):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.expm1(np.abs(values))


def make_mape_sample_weights(y, clip_denominator):
    y = np.asarray(y, dtype=float).reshape(-1)
    max_weight = 1.0 / float(clip_denominator)
    weights = np.clip(1.0 / (np.abs(y) + 1e-5), 0.0, max_weight)
    mean_weight = float(np.mean(weights))
    if mean_weight <= 0.0:
        raise ValueError("Sample weights must have positive mean.")
    return weights / mean_weight


def parse_max_depth(raw_depth):
    raw_depth = raw_depth.strip().lower()
    if raw_depth in ("none", "null", ""):
        return None
    return int(raw_depth)


def parse_rf_configs(raw_configs):
    if raw_configs is None:
        raw_configs = ";".join(DEFAULT_RF_CONFIGS)
    configs = []
    for raw_config in raw_configs.split(";"):
        raw_config = raw_config.strip()
        if not raw_config:
            continue
        parts = [part.strip() for part in raw_config.split(":")]
        if len(parts) != 6:
            raise ValueError("RF config must use name:n_estimators:min_samples_leaf:max_depth:seed:clip_denominator.")
        name, n_estimators, min_samples_leaf, max_depth, seed, clip_denominator = parts
        configs.append(
            RFConfig(
                name=name,
                n_estimators=int(n_estimators),
                min_samples_leaf=int(min_samples_leaf),
                max_depth=parse_max_depth(max_depth),
                random_state=int(seed),
                clip_denominator=float(clip_denominator),
            )
        )
    if not configs:
        raise ValueError("At least one RF config is required.")
    names = [config.name for config in configs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate RF config name(s): {', '.join(duplicates)}")
    return configs


def parse_seed_list(raw_seeds):
    if raw_seeds is None:
        return list(DEFAULT_STRATIFIED_SEEDS)
    seeds = [int(seed.strip()) for seed in raw_seeds.split(",") if seed.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def validate_output_folder_available(output_folder, overwrite=False):
    output_path = Path(output_folder)
    if output_path.exists() and any(output_path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output folder already exists and is not empty: {output_path}. "
            "Choose a new --output-folder or pass --overwrite."
        )


def stack_cases(cases):
    return np.vstack([case["X"] for case in cases]), np.vstack([case["y"] for case in cases]).reshape(-1)


def overpressure_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-5))) * 100.0)


def make_model(config):
    return RandomForestRegressor(
        n_estimators=config.n_estimators,
        min_samples_leaf=config.min_samples_leaf,
        max_depth=config.max_depth,
        max_features=1.0,
        bootstrap=True,
        n_jobs=1,
        random_state=config.random_state,
    )


def generate_stratified_case_folds(all_cases, n_folds, seed):
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if len(all_cases) < n_folds:
        raise ValueError("At least n_folds cases are required.")

    buckets = {}
    for case in all_cases:
        buckets.setdefault(case_difficulty_bucket(case), []).append(case)

    rng = np.random.default_rng(seed)
    fold_test_cases = [[] for _ in range(n_folds)]
    for bucket_name in sorted(buckets):
        bucket_cases = sorted(buckets[bucket_name], key=lambda case: case["case_file"])
        indices = np.arange(len(bucket_cases))
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            fold_test_cases[position % n_folds].append(bucket_cases[int(index)])

    case_by_name = {case["case_file"]: case for case in all_cases}
    all_names = set(case_by_name)
    folds = []
    for test_cases in fold_test_cases:
        test_names = {case["case_file"] for case in test_cases}
        train_names = sorted(all_names - test_names)
        folds.append(
            CaseSplit(
                train_cases=[dict(case_by_name[name], split="train") for name in train_names],
                validation_cases=[],
                test_cases=[
                    dict(case_by_name[name], split="test")
                    for name in sorted(test_names)
                ],
            )
        )
    return folds


def train_predict_config(config, train_cases, test_cases):
    X_train, y_train = stack_cases(train_cases)
    X_test, y_test = stack_cases(test_cases)
    weights = make_mape_sample_weights(y_train, config.clip_denominator)
    model = make_model(config)
    model.fit(X_train, signed_log_target(y_train), sample_weight=weights)
    pred_test = inverse_signed_log_target(model.predict(X_test))
    return model, {
        "test_mape": overpressure_mape(y_test, pred_test),
        "test_r2": float(r2_score(y_test, pred_test)),
    }


def train_predict_mlp_baseline(train_cases, test_cases, args, run_id, output_root=None):
    output_root = Path(output_root or args.output_folder)
    seed = args.baseline_seed + run_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    scaler_X, train_loader, _test_loader, train_points, test_points = make_loaders(
        train_cases,
        test_cases,
        args.batch_size,
        seed,
    )

    model = ImprovedMLP(input_dim=10, dropout_p=args.baseline_dropout)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.baseline_lr,
        weight_decay=args.baseline_weight_decay,
    )
    print(
        f"baseline run={run_id} train_points={train_points} test_points={test_points} "
        f"epochs={args.epochs} seed={seed}"
    )
    for _epoch in range(args.epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = mape_loss(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    model_dir = output_root / "baseline_model" / f"run_{run_id}"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pth")
    joblib.dump(scaler_X, model_dir / "scaler_X.pkl")

    write_mlp_predictions(model, scaler_X, train_cases, output_root / "baseline_predictions" / f"run_{run_id}" / "train")
    write_mlp_predictions(model, scaler_X, test_cases, output_root / "baseline_predictions" / f"run_{run_id}" / "test")

    X_test, y_test = stack_cases(test_cases)
    pred_abs = predict_member(model, scaler_X, X_test).reshape(-1)
    pred_over = pred_abs - ATM_PRESSURE
    return {
        "test_mape": overpressure_mape(y_test, pred_over),
        "test_r2": float(r2_score(y_test, pred_over)),
    }


def write_mlp_predictions(model, scaler_X, cases, output_dir):
    for case in cases:
        preds_abs = predict_member(model, scaler_X, case["X"]).reshape(-1)
        preds_var = np.zeros_like(preds_abs)
        write_prediction_file(Path(output_dir) / case["case_file"], case["raw_points"], preds_abs, preds_var)


def write_case_predictions(model, cases, output_dir):
    for case in cases:
        preds_over = inverse_signed_log_target(model.predict(case["X"]))
        preds_abs = preds_over + 101325.0
        preds_var = np.zeros_like(preds_abs)
        write_prediction_file(Path(output_dir) / case["case_file"], case["raw_points"], preds_abs, preds_var)


def write_rows(rows, output_path):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_formal_manifest(folds, output_path):
    rows = []
    for fold_index, fold in enumerate(folds, start=1):
        manifest = split_manifest(fold)
        for split in ("train", "validation", "test"):
            for case_file in manifest[split]:
                rows.append({"fold": fold_index, "split": split, "case_file": case_file})
    write_rows(rows, output_path)


def summarize_results(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["config"], []).append(row)
    summary_rows = []
    for config, config_rows in sorted(grouped.items()):
        mapes = [float(row["test_mape"]) for row in config_rows]
        r2s = [float(row["test_r2"]) for row in config_rows]
        summary_rows.append(
            {
                "config": config,
                "folds": len(config_rows),
                "avg_test_mape": float(np.mean(mapes)),
                "std_test_mape": float(np.std(mapes)),
                "avg_test_r2": float(np.mean(r2s)),
                "per_fold_mape": ";".join(f"{mape:.4f}" for mape in mapes),
            }
        )
    return sorted(summary_rows, key=lambda row: row["avg_test_mape"])


def _mean(values):
    return float(np.mean([float(value) for value in values]))


def _std(values):
    return float(np.std([float(value) for value in values]))


def summarize_formal_pair_rows(rows, threshold=15.0):
    if not rows:
        raise ValueError("No formal pair rows available for summary.")

    baseline_mapes = [float(row["baseline_mape"]) for row in rows]
    baseline_r2s = [float(row["baseline_r2"]) for row in rows]
    candidate_mapes = [float(row["candidate_mape"]) for row in rows]
    candidate_r2s = [float(row["candidate_r2"]) for row in rows]
    deltas = [candidate - baseline for candidate, baseline in zip(candidate_mapes, baseline_mapes)]
    threshold = float(threshold)
    return {
        "folds": len(rows),
        "baseline_avg_mape": _mean(baseline_mapes),
        "baseline_std_mape": _std(baseline_mapes),
        "baseline_avg_r2": _mean(baseline_r2s),
        "baseline_std_r2": _std(baseline_r2s),
        "candidate_avg_mape": _mean(candidate_mapes),
        "candidate_std_mape": _std(candidate_mapes),
        "candidate_avg_r2": _mean(candidate_r2s),
        "candidate_std_r2": _std(candidate_r2s),
        "delta_avg_mape_candidate_minus_baseline": _mean(deltas),
        "delta_std_mape_candidate_minus_baseline": _std(deltas),
        "threshold_percent": threshold,
        "candidate_avg_below_15_percent": _mean(candidate_mapes) < threshold,
        "candidate_all_folds_below_15_percent": all(mape < threshold for mape in candidate_mapes),
        "candidate_folds_below_15_percent": sum(1 for mape in candidate_mapes if mape < threshold),
    }


def write_frozen_input_stratification_rule(output_path):
    rows = [
        {
            "rule_name": "low_charge_near",
            "condition": "median(blast)<=1 and median(Z)<2500 and max(Z)<6000",
            "allowed_inputs": "blast;case median Z;case max Z",
        },
        {
            "rule_name": "low_charge_far",
            "condition": "median(blast)<=1 and (median(Z)>=2500 or max(Z)>=6000)",
            "allowed_inputs": "blast;case median Z;case max Z",
        },
        {
            "rule_name": "mid_charge_near",
            "condition": "1<median(blast)<100 and median(Z)<1000",
            "allowed_inputs": "blast;case median Z",
        },
        {
            "rule_name": "mid_charge_far",
            "condition": "1<median(blast)<100 and median(Z)>=1000",
            "allowed_inputs": "blast;case median Z",
        },
        {
            "rule_name": "high_charge_near",
            "condition": "median(blast)>=100 and median(Z)<250",
            "allowed_inputs": "blast;case median Z",
        },
        {
            "rule_name": "high_charge_far",
            "condition": "median(blast)>=100 and median(Z)>=250",
            "allowed_inputs": "blast;case median Z",
        },
    ]
    write_rows(rows, output_path)


def run_stratified(args, configs, all_cases):
    seeds = parse_seed_list(args.stratified_seeds)
    rows = []
    for fold_index, seed in enumerate(seeds, start=1):
        case_split = generate_stratified_case_split(
            all_cases,
            test_fraction=args.test_fraction,
            validation_fraction=args.validation_fraction,
            seed=seed,
        )
        train_cases = case_split.train_cases + case_split.validation_cases
        test_cases = case_split.test_cases
        manifest = split_manifest(case_split)
        fold_dir = Path(args.output_folder) / "stratified" / f"seed_{seed}"
        write_screening_manifest(manifest, fold_dir / "split_manifest.csv")
        for config in configs:
            model, metrics = train_predict_config(config, train_cases, test_cases)
            prediction_dir = fold_dir / "predictions" / config.name / "run_1" / "test"
            write_case_predictions(model, test_cases, prediction_dir)
            rows.append(
                {
                    "scheme": "stratified",
                    "fold": fold_index,
                    "seed": seed,
                    "config": config.name,
                    "test_mape": metrics["test_mape"],
                    "test_r2": metrics["test_r2"],
                    "test_cases": ";".join(manifest["test"]),
                }
            )
            print(
                f"stratified seed={seed} config={config.name} "
                f"MAPE={metrics['test_mape']:.4f}% R2={metrics['test_r2']:.4f}"
            )
    write_rows(rows, Path(args.output_folder) / "stratified_results.csv")
    summary_rows = summarize_results(rows)
    write_rows(summary_rows, Path(args.output_folder) / "stratified_summary.csv")
    return rows, summary_rows


def run_legacy_fivefold(args, configs, all_cases):
    rows = []
    for run_id in range(1, args.num_runs + 1):
        manifest = load_split_manifest(args.baseline_pred_dir, run_id)
        train_cases, test_cases = cases_from_manifest(all_cases, manifest)
        fold_dir = Path(args.output_folder) / "legacy_fivefold" / f"run_{run_id}"
        write_screening_manifest(
            {"train": manifest["train"], "validation": [], "test": manifest["test"]},
            fold_dir / "split_manifest.csv",
        )
        for config in configs:
            model, metrics = train_predict_config(config, train_cases, test_cases)
            prediction_dir = fold_dir / "predictions" / config.name / f"run_{run_id}" / "test"
            write_case_predictions(model, test_cases, prediction_dir)
            rows.append(
                {
                    "scheme": "legacy_fivefold",
                    "fold": run_id,
                    "seed": "",
                    "config": config.name,
                    "test_mape": metrics["test_mape"],
                    "test_r2": metrics["test_r2"],
                    "test_cases": ";".join(manifest["test"]),
                }
            )
            print(
                f"legacy run={run_id} config={config.name} "
                f"MAPE={metrics['test_mape']:.4f}% R2={metrics['test_r2']:.4f}"
            )
    write_rows(rows, Path(args.output_folder) / "legacy_fivefold_results.csv")
    summary_rows = summarize_results(rows)
    write_rows(summary_rows, Path(args.output_folder) / "legacy_fivefold_summary.csv")
    return rows, summary_rows


def run_formal_fivefold(args, configs, all_cases):
    if len(configs) != 1:
        raise ValueError("Formal review must freeze exactly one RF candidate config.")
    config = configs[0]
    if config.name != args.formal_candidate_config:
        raise ValueError(
            f"Formal candidate config is frozen as {args.formal_candidate_config}, got {config.name}."
        )

    output_root = Path(args.output_folder) / "formal_fivefold"
    folds = generate_stratified_case_folds(all_cases, args.formal_folds, args.formal_seed)
    write_formal_manifest(folds, output_root / "formal_manifest.csv")
    write_frozen_input_stratification_rule(output_root / "frozen_input_stratification_rule.csv")

    rows = []
    for run_id, fold in enumerate(folds, start=1):
        manifest = split_manifest(fold)
        fold_dir = output_root / "manifests" / f"run_{run_id}"
        write_screening_manifest(manifest, fold_dir / "split_manifest.csv")

        baseline_metrics = train_predict_mlp_baseline(
            fold.train_cases,
            fold.test_cases,
            args,
            run_id,
            output_root=output_root,
        )
        rf_model, candidate_metrics = train_predict_config(config, fold.train_cases, fold.test_cases)
        rf_train_dir = output_root / "candidate_predictions" / config.name / f"run_{run_id}" / "train"
        rf_test_dir = output_root / "candidate_predictions" / config.name / f"run_{run_id}" / "test"
        write_case_predictions(rf_model, fold.train_cases, rf_train_dir)
        write_case_predictions(rf_model, fold.test_cases, rf_test_dir)

        row = {
            "fold": run_id,
            "manifest_seed": args.formal_seed,
            "baseline_model": "original_mlp_baseline",
            "candidate_config": config.name,
            "baseline_mape": baseline_metrics["test_mape"],
            "baseline_r2": baseline_metrics["test_r2"],
            "candidate_mape": candidate_metrics["test_mape"],
            "candidate_r2": candidate_metrics["test_r2"],
            "delta_mape_candidate_minus_baseline": candidate_metrics["test_mape"] - baseline_metrics["test_mape"],
            "candidate_below_15_percent": candidate_metrics["test_mape"] < args.threshold_percent,
            "test_cases": ";".join(manifest["test"]),
        }
        rows.append(row)
        print(
            f"formal fold={run_id} baseline MAPE={baseline_metrics['test_mape']:.4f}% "
            f"R2={baseline_metrics['test_r2']:.4f} | {config.name} "
            f"MAPE={candidate_metrics['test_mape']:.4f}% R2={candidate_metrics['test_r2']:.4f}"
        )

    summary = summarize_formal_pair_rows(rows, threshold=args.threshold_percent)
    summary.update(
        {
            "manifest_seed": args.formal_seed,
            "stratification_rule": "input_feature_case_bucket_v1",
            "candidate_config": config.name,
            "baseline_model": "original_mlp_baseline",
        }
    )
    write_rows(rows, output_root / "formal_pair_results.csv")
    write_rows([summary], output_root / "formal_summary.csv")
    return rows, summary


def write_repro_commands(args):
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    if args.formal_fivefold:
        command = (
            f"python {Path(__file__).name}"
            f" --formal-fivefold"
            f" --output-folder {args.output_folder}"
            f" --data-folder {args.data_folder}"
            f" --case-info-path {args.case_info_path}"
            f" --rf-configs \"{args.rf_configs or FORMAL_RF_CONFIG}\""
            f" --formal-candidate-config {args.formal_candidate_config}"
            f" --formal-folds {args.formal_folds}"
            f" --formal-seed {args.formal_seed}"
            f" --epochs {args.epochs}"
            f" --batch-size {args.batch_size}"
            f" --baseline-seed {args.baseline_seed}"
            f" --threshold-percent {args.threshold_percent}"
        )
        rerun_note = (
            "\n# To preserve this artifact, rerun with a new --output-folder.\n"
            "# To intentionally replace this exact output folder, add --overwrite.\n"
        )
    else:
        command = (
            f"python {Path(__file__).name}"
            f" --output-folder {args.output_folder}"
            f" --data-folder {args.data_folder}"
            f" --case-info-path {args.case_info_path}"
            f" --test-fraction {args.test_fraction}"
            f" --validation-fraction {args.validation_fraction}"
            f" --stratified-seeds {args.stratified_seeds or ','.join(str(seed) for seed in DEFAULT_STRATIFIED_SEEDS)}"
        )
        if args.rf_configs:
            command += f' --rf-configs "{args.rf_configs}"'
        if args.legacy_fivefold:
            command += " --legacy-fivefold"
        rerun_note = ""
    with open(output_folder / "repro_commands.txt", "w", encoding="utf-8") as f:
        f.write(command + "\n")
        f.write(rerun_note)


def parse_args():
    parser = argparse.ArgumentParser(description="Random forest signed-log MAPE screening.")
    parser.add_argument("--output-folder", default=None)
    parser.add_argument("--data-folder", default=DATA_FOLDER)
    parser.add_argument("--case-info-path", default="data/case_info.csv")
    parser.add_argument("--rf-configs", default=None)
    parser.add_argument("--stratified-seeds", default=None)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--legacy-fivefold", action="store_true")
    parser.add_argument("--baseline-pred-dir", default="predictions")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--formal-fivefold", action="store_true")
    parser.add_argument("--formal-folds", type=int, default=5)
    parser.add_argument("--formal-seed", type=int, default=20260612)
    parser.add_argument("--formal-candidate-config", default=DEFAULT_FORMAL_CANDIDATE_CONFIG)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--baseline-lr", type=float, default=LR)
    parser.add_argument("--baseline-dropout", type=float, default=0.0)
    parser.add_argument("--baseline-weight-decay", type=float, default=0.0)
    parser.add_argument("--baseline-seed", type=int, default=42)
    parser.add_argument("--threshold-percent", type=float, default=15.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output_folder is None:
        args.output_folder = FORMAL_OUTPUT_FOLDER if args.formal_fivefold else OUTPUT_FOLDER
    if args.formal_fivefold and args.rf_configs is None:
        args.rf_configs = FORMAL_RF_CONFIG
    return args


def main():
    args = parse_args()
    if args.formal_fivefold:
        validate_output_folder_available(args.output_folder, overwrite=args.overwrite)
    write_repro_commands(args)
    configs = parse_rf_configs(args.rf_configs)
    all_cases = load_all_case_data(args.data_folder, args.case_info_path)
    if not all_cases:
        raise RuntimeError(f"No valid case data loaded from {args.data_folder}")
    if args.formal_fivefold:
        run_formal_fivefold(args, configs, all_cases)
    else:
        run_stratified(args, configs, all_cases)
        if args.legacy_fivefold:
            run_legacy_fivefold(args, configs, all_cases)


if __name__ == "__main__":
    main()
