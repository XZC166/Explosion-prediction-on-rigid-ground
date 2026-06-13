import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ATM_PRESSURE = 101325.0
DEFAULT_PRED_BASE = (
    "ensemble_outputs/mape_rf_screening_formal_stratified_fivefold/"
    "formal_fivefold/candidate_predictions/rf200_l2_d16_clip5e4"
)
DEFAULT_TRUE_DIR = "data/collect_pressure_peak"
DEFAULT_OUTPUT_DIR = "rf_visual_results"
DEFAULT_INDIVIDUAL_CASES = ["value14", "value18", "value26", "value31", "value36", "value5"]


@dataclass
class RfPredictionData:
    y_true: np.ndarray
    y_pred: np.ndarray
    case_results: pd.DataFrame


def get_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.size"] = 16
    return plt


def run_sort_key(path):
    match = re.fullmatch(r"run_(\d+)", path.parent.name)
    return int(match.group(1)) if match else path.parent.name


def discover_prediction_dirs(pred_base, split="test"):
    pred_base = Path(pred_base)
    if not pred_base.exists():
        raise FileNotFoundError(f"Prediction folder does not exist: {pred_base}")

    dirs = []
    for run_dir in pred_base.iterdir():
        if not run_dir.is_dir() or not re.fullmatch(r"run_\d+", run_dir.name):
            continue
        split_dir = run_dir / split
        if split_dir.is_dir():
            dirs.append(split_dir)
    return sorted(dirs, key=run_sort_key)


def calculate_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        return {"MAE": np.nan, "MAPE": np.nan, "RMSE": np.nan, "R2": np.nan}

    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-10))) * 100
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    r2 = r2_score(y_true_clean, y_pred_clean) if len(y_true_clean) >= 2 else np.nan
    return {"MAE": mae, "MAPE": mape, "RMSE": rmse, "R2": r2}


def load_array(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def load_rf_predictions(pred_base, true_dir, split="test", atm_pressure=ATM_PRESSURE):
    true_dir = Path(true_dir)
    if not true_dir.exists():
        raise FileNotFoundError(f"True data folder does not exist: {true_dir}")

    all_y_true = []
    all_y_pred = []
    rows = []

    for pred_dir in discover_prediction_dirs(pred_base, split=split):
        fold_match = re.fullmatch(r"run_(\d+)", pred_dir.parent.name)
        fold = int(fold_match.group(1)) if fold_match else np.nan

        for pred_file in sorted(path for path in pred_dir.iterdir() if path.is_file()):
            true_file = true_dir / pred_file.name
            if not true_file.exists():
                print(f"Warning: missing true file for {pred_file.name}; skipped.")
                continue

            pred_data = load_array(pred_file)
            true_data = load_array(true_file)
            if pred_data.shape[1] < 4 or true_data.shape[1] < 4:
                print(f"Warning: {pred_file.name} needs at least 4 columns; skipped.")
                continue

            n_samples = min(len(pred_data), len(true_data))
            if len(pred_data) != len(true_data):
                print(
                    f"Warning: {pred_file.name} length mismatch "
                    f"pred={len(pred_data)} true={len(true_data)}; truncated to {n_samples}."
                )

            y_pred = pred_data[:n_samples, 3] - atm_pressure
            y_true = true_data[:n_samples, 3] - atm_pressure
            metrics = calculate_metrics(y_true, y_pred)

            rows.append(
                {
                    "fold": fold,
                    "case_file": pred_file.name,
                    "n_samples": n_samples,
                    "MAE": metrics["MAE"],
                    "MAPE": metrics["MAPE"],
                    "RMSE": metrics["RMSE"],
                    "R2": metrics["R2"],
                }
            )
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)

    if not rows:
        raise ValueError("No matched RF prediction/true data files were loaded.")

    return RfPredictionData(
        y_true=np.asarray(all_y_true, dtype=float),
        y_pred=np.asarray(all_y_pred, dtype=float),
        case_results=pd.DataFrame(rows),
    )


def iter_matched_prediction_arrays(pred_base, true_dir, split="test"):
    true_dir = Path(true_dir)
    if not true_dir.exists():
        raise FileNotFoundError(f"True data folder does not exist: {true_dir}")

    for pred_dir in discover_prediction_dirs(pred_base, split=split):
        fold_match = re.fullmatch(r"run_(\d+)", pred_dir.parent.name)
        fold = int(fold_match.group(1)) if fold_match else np.nan

        for pred_file in sorted(path for path in pred_dir.iterdir() if path.is_file()):
            true_file = true_dir / pred_file.name
            if not true_file.exists():
                print(f"Warning: missing true file for {pred_file.name}; skipped.")
                continue

            pred_data = load_array(pred_file)
            true_data = load_array(true_file)
            if pred_data.shape[1] < 4 or true_data.shape[1] < 4:
                print(f"Warning: {pred_file.name} needs at least 4 columns; skipped.")
                continue

            n_samples = min(len(pred_data), len(true_data))
            if len(pred_data) != len(true_data):
                print(
                    f"Warning: {pred_file.name} length mismatch "
                    f"pred={len(pred_data)} true={len(true_data)}; truncated to {n_samples}."
                )

            yield fold, pred_file.name, pred_data[:n_samples], true_data[:n_samples]


def build_point_dataframe(pred_base, true_dir, split="test", atm_pressure=ATM_PRESSURE):
    rows = []
    for fold, case_file, pred_data, true_data in iter_matched_prediction_arrays(pred_base, true_dir, split=split):
        x = true_data[:, 0]
        y = true_data[:, 1]
        z = true_data[:, 2]
        distance = np.sqrt(x**2 + y**2 + z**2)
        true_kpa = (true_data[:, 3] - atm_pressure) / 1000
        pred_kpa = (pred_data[:, 3] - atm_pressure) / 1000

        for values in zip(x, y, z, distance, true_kpa, pred_kpa):
            rows.append(
                {
                    "fold": fold,
                    "case_file": case_file,
                    "x": values[0],
                    "y": values[1],
                    "z": values[2],
                    "distance": values[3],
                    "true_kpa": values[4],
                    "pred_kpa": values[5],
                }
            )

    if not rows:
        raise ValueError("No matched RF prediction/true data files were loaded.")

    return pd.DataFrame(rows, columns=["fold", "case_file", "x", "y", "z", "distance", "true_kpa", "pred_kpa"])


def normalize_case_name(case_name):
    raw = str(case_name).strip()
    if raw.startswith("value"):
        return raw
    return f"value{raw}"


def filter_case_points(points_df, case_names):
    normalized_names = [normalize_case_name(name) for name in case_names]
    return points_df[points_df["case_file"].isin(normalized_names)].copy()


def plot_predictions_scatter(y_true, y_pred, save_path=None, figsize=(8, 8)):
    plt = get_pyplot()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask] / 1000
    y_pred_clean = y_pred[mask] / 1000

    if len(y_true_clean) == 0:
        print("Error: no valid data points.")
        return

    max_val = max(y_true_clean.max(), y_pred_clean.max())
    min_val = min(y_true_clean.min(), y_pred_clean.min())
    margin = (max_val - min_val) * 0.05
    plot_min = min_val - margin
    plot_max = max_val + margin

    plt.figure(figsize=figsize)
    plt.scatter(y_true_clean, y_pred_clean, alpha=0.5, s=20, c="blue", edgecolors="none")
    plt.plot([plot_min, plot_max], [plot_min, plot_max], "r", linewidth=2, label="Perfect Prediction (y=x)")
    plt.plot([plot_min, plot_max], [plot_min * 1.3, plot_max * 1.3], "r--", linewidth=1.5, alpha=0.8, label="+30% Error")
    plt.plot([plot_min, plot_max], [plot_min * 0.7, plot_max * 0.7], "r--", linewidth=1.5, alpha=0.8, label="-30% Error")
    plt.fill_between(
        [plot_min, plot_max],
        [plot_min * 0.7, plot_max * 0.7],
        [plot_min * 1.3, plot_max * 1.3],
        alpha=0.1,
        color="red",
        label="±30% Error Band",
    )
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)
    plt.xlabel("True(kPa)", fontsize=16)
    plt.ylabel("Pred(kPa)", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Scatter plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_prediction_magnitude_comparison(y_true, y_pred, save_path=None, figsize=(12, 6)):
    plt = get_pyplot()
    mask_true = ~np.isnan(y_true)
    mask_pred = ~np.isnan(y_pred)
    y_true_clean = y_true[mask_true]
    y_pred_clean = y_pred[mask_pred]

    bins = [0, 100, 1000, 10000, 100000, 1000000, np.inf]
    labels = ["0-100 Pa", "100-1000 Pa", "1000-10 kPa", "10-100 kPa", "100-1000 kPa", ">1000 kPa"]

    counts_true, _ = np.histogram(y_true_clean, bins=bins)
    counts_pred, _ = np.histogram(y_pred_clean, bins=bins)
    percentages_true = counts_true / len(y_true_clean) * 100
    percentages_pred = counts_pred / len(y_pred_clean) * 100

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(
        x - width / 2,
        percentages_true,
        width,
        label="True Values",
        color="steelblue",
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )
    bars2 = ax.bar(
        x + width / 2,
        percentages_pred,
        width,
        label="Predicted Values",
        color="coral",
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    ax.set_xlabel("OverPressure Range", fontsize=14)
    ax.set_ylabel("Percentage (%)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(loc="upper left", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    for bars, percentages in zip([bars1, bars2], [percentages_true, percentages_pred]):
        for bar, pct in zip(bars, percentages):
            if pct > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Magnitude comparison plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_mape_distribution(y_true, y_pred, save_path=None, figsize=(12, 6)):
    plt = get_pyplot()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        print("Error: no valid data points.")
        return

    mape = np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-10)) * 100
    mape_bins = [0, 5, 10, 15, 20, 30, np.inf]
    mape_labels = ["0-5%", "5-10%", "10-15%", "15-20%", "20-30%", ">30%"]
    hist, _ = np.histogram(mape, bins=mape_bins)
    percentages = hist / len(mape) * 100

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e67e22", "#e74c3c", "#c0392b"]
    bars = ax.bar(mape_labels, percentages, color=colors, alpha=0.8, edgecolor="black", linewidth=1)

    for bar, pct in zip(bars, percentages):
        if pct > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

    ax.set_xlabel("MAPE Range", fontsize=16)
    ax.set_ylabel("Percentage (%)", fontsize=16)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"MAPE distribution plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_mape_distribution_by_magnitude(y_true, y_pred, save_path=None, figsize=(12, 6)):
    plt = get_pyplot()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) == 0:
        print("Error: no valid data points.")
        return

    mape = np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-10)) * 100
    pressure_bins = [0, 1000, 10000, 100000, 1000000, np.inf]
    pressure_labels = ["0-1 kPa", "1-10 kPa", "10-100 kPa", "100-1000 kPa", ">1000 kPa"]
    mape_bins = [0, 5, 10, 15, 20, 30, np.inf]
    mape_labels = ["0-5%", "5-10%", "10-15%", "15-20%", "20-30%", ">30%"]

    distribution_data = []
    pressure_counts = []
    for i in range(len(pressure_bins) - 1):
        pressure_mask = (y_true_clean >= pressure_bins[i]) & (y_true_clean < pressure_bins[i + 1])
        mape_in_range = mape[pressure_mask]
        count_in_range = np.sum(pressure_mask)
        pressure_counts.append(count_in_range)
        if count_in_range > 0:
            hist, _ = np.histogram(mape_in_range, bins=mape_bins)
            distribution_data.append(hist / count_in_range * 100)
        else:
            distribution_data.append(np.zeros(len(mape_labels)))

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(mape_labels))
    width = 0.15
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]

    for i, (pressure_label, color) in enumerate(zip(pressure_labels, colors)):
        percentages = distribution_data[i]
        bars = ax.bar(
            x + i * width,
            percentages,
            width,
            label=pressure_label,
            color=color,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, pct in zip(bars, percentages):
            if pct > 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

    ax.set_xlabel("MAPE Range", fontsize=16)
    ax.set_ylabel("Percentage (%)", fontsize=16)
    ax.set_title("MAPE Distribution by Pressure Magnitude", fontsize=16)
    ax.set_xticks(x + width * (len(pressure_labels) - 1) / 2)
    ax.set_xticklabels(mape_labels)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=16, title="Pressure Range")
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"MAPE by magnitude plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()

    return {
        "pressure_ranges": pressure_labels,
        "pressure_counts": pressure_counts,
        "mape_ranges": mape_labels,
        "distribution": distribution_data,
    }


def plot_rf_all_cases_combined(points_df, save_path=None, figsize=(15, 7)):
    plt = get_pyplot()
    df_sorted = points_df.sort_values(by="true_kpa").reset_index(drop=True)
    x_axis = np.arange(len(df_sorted))

    plt.figure(figsize=figsize)
    plt.plot(x_axis, df_sorted["pred_kpa"], "g-", label="Pred", linewidth=1.5)
    plt.scatter(x_axis, df_sorted["true_kpa"], color="black", s=2, label="True", alpha=0.6)
    plt.xlabel("POI")
    plt.ylabel("OverPressure(kPa)")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"RF combined POI plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def binned_distance_summary(points_df, n_bins=40):
    df = points_df.sort_values(by="distance").reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No points available for distance summary.")

    if df["distance"].min() == df["distance"].max():
        bins = np.array([df["distance"].min(), df["distance"].max() + 1e-9])
    else:
        bins = np.linspace(df["distance"].min(), df["distance"].max(), n_bins + 1)

    rows = []
    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            mask = (df["distance"] >= bins[i]) & (df["distance"] <= bins[i + 1])
        else:
            mask = (df["distance"] >= bins[i]) & (df["distance"] < bins[i + 1])
        bin_df = df[mask]
        if bin_df.empty:
            continue

        true_mean = bin_df["true_kpa"].mean()
        pred_mean = bin_df["pred_kpa"].mean()
        residual = bin_df["pred_kpa"] - bin_df["true_kpa"]
        residual_std = residual.std(ddof=0)
        rows.append(
            {
                "distance": bin_df["distance"].mean(),
                "true_kpa": true_mean,
                "pred_kpa": pred_mean,
                "lower_kpa": pred_mean - residual_std,
                "upper_kpa": pred_mean + residual_std,
                "count": len(bin_df),
            }
        )

    return pd.DataFrame(rows)


def plot_rf_distance_comparison(points_df, save_path=None, figsize=(10, 6), n_bins=40):
    plt = get_pyplot()
    summary = binned_distance_summary(points_df, n_bins=n_bins)

    plt.figure(figsize=figsize)
    plt.plot(summary["distance"], summary["true_kpa"], "k--", label="True", alpha=0.7)
    plt.plot(summary["distance"], summary["pred_kpa"], "r-", label="Pred", linewidth=2)
    plt.fill_between(
        summary["distance"],
        summary["lower_kpa"],
        summary["upper_kpa"],
        color="red",
        alpha=0.15,
        label="Binned variation band",
    )
    plt.xlabel("Distance (m)")
    plt.ylabel("OverPressure(kPa)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"RF distance comparison plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()

    return summary


def single_case_plot_series(case_df, y_scale=1000.0):
    if case_df.empty:
        raise ValueError("No points available for the selected case.")

    df_sorted = case_df.sort_values(by="distance").reset_index(drop=True)
    residual = df_sorted["pred_kpa"] - df_sorted["true_kpa"]
    residual_std = residual.std(ddof=0)
    return {
        "distance": df_sorted["distance"].to_numpy(),
        "true": df_sorted["true_kpa"].to_numpy() * y_scale,
        "pred": df_sorted["pred_kpa"].to_numpy() * y_scale,
        "lower": (df_sorted["pred_kpa"].to_numpy() - residual_std) * y_scale,
        "upper": (df_sorted["pred_kpa"].to_numpy() + residual_std) * y_scale,
    }


def plot_single_case_distance_comparison(case_df, save_path=None, figsize=(10, 6), y_scale=1000.0, y_label="OverPressure(Pa)"):
    plt = get_pyplot()
    series = single_case_plot_series(case_df, y_scale=y_scale)

    plt.figure(figsize=figsize)
    plt.plot(series["distance"], series["true"], "k--", label="True", alpha=0.7)
    plt.plot(series["distance"], series["pred"], "r-", label="Pred", linewidth=2)
    plt.fill_between(
        series["distance"],
        series["lower"],
        series["upper"],
        color="red",
        alpha=0.15,
        label="Case residual band",
    )
    plt.xlabel("Distance (m)")
    plt.ylabel(y_label)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"RF single-case distance plot saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_selected_case_distance_comparisons(points_df, case_names, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for case_name in [normalize_case_name(name) for name in case_names]:
        case_df = points_df[points_df["case_file"] == case_name]
        if case_df.empty:
            print(f"Warning: {case_name} is not present in the loaded RF data; skipped.")
            continue
        save_path = output_dir / f"{case_name}_rf_distance_comparison.png"
        plot_single_case_distance_comparison(case_df, save_path=save_path)
        written.append(save_path)
    return written


def write_outputs(data, output_dir, points_df=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overall_metrics = calculate_metrics(data.y_true, data.y_pred)
    plot_predictions_scatter(data.y_true, data.y_pred, output_dir / "scatter.png")
    plot_prediction_magnitude_comparison(data.y_true, data.y_pred, output_dir / "true_vs_pred_magnitude.png")
    plot_mape_distribution_by_magnitude(data.y_true, data.y_pred, output_dir / "mape_distribution_by_magnitude.png")
    plot_mape_distribution(data.y_true, data.y_pred, output_dir / "mape_distribution.png")
    if points_df is not None:
        plot_rf_all_cases_combined(points_df, output_dir / "rf_all_cases_combined.png")
        distance_summary = plot_rf_distance_comparison(points_df, output_dir / "rf_distance_comparison.png")
        distance_summary.to_csv(output_dir / "rf_distance_summary.csv", index=False)
        plot_selected_case_distance_comparisons(
            points_df,
            DEFAULT_INDIVIDUAL_CASES,
            output_dir / "individual_cases",
        )

    data.case_results.to_csv(output_dir / "rf_case_results.csv", index=False)
    summary_df = pd.DataFrame(
        {
            "Metric": ["MAE", "MAPE", "RMSE", "R2"],
            "Overall": [
                overall_metrics["MAE"],
                overall_metrics["MAPE"],
                overall_metrics["RMSE"],
                overall_metrics["R2"],
            ],
            "MeanByCase": [
                data.case_results["MAE"].mean(),
                data.case_results["MAPE"].mean(),
                data.case_results["RMSE"].mean(),
                data.case_results["R2"].mean(),
            ],
            "StdByCase": [
                data.case_results["MAE"].std(),
                data.case_results["MAPE"].std(),
                data.case_results["RMSE"].std(),
                data.case_results["R2"].std(),
            ],
        }
    )
    summary_df.to_csv(output_dir / "rf_summary.csv", index=False)
    return overall_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize final RF prediction results.")
    parser.add_argument("--pred-base", default=DEFAULT_PRED_BASE)
    parser.add_argument("--true-dir", default=DEFAULT_TRUE_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--atm-pressure", type=float, default=ATM_PRESSURE)
    parser.add_argument(
        "--individual-cases",
        default=",".join(DEFAULT_INDIVIDUAL_CASES),
        help="Comma-separated case names or numbers for individual distance plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_rf_predictions(
        args.pred_base,
        args.true_dir,
        split=args.split,
        atm_pressure=args.atm_pressure,
    )
    points_df = build_point_dataframe(
        args.pred_base,
        args.true_dir,
        split=args.split,
        atm_pressure=args.atm_pressure,
    )
    global DEFAULT_INDIVIDUAL_CASES
    DEFAULT_INDIVIDUAL_CASES = [case.strip() for case in args.individual_cases.split(",") if case.strip()]
    metrics = write_outputs(data, args.output_dir, points_df=points_df)

    print("\n" + "=" * 80)
    print("Final RF visualization summary")
    print("=" * 80)
    print(f"Prediction folder: {args.pred_base}")
    print(f"True data folder: {args.true_dir}")
    print(f"Output folder: {args.output_dir}")
    print(f"Total samples: {len(data.y_true)}")
    print(f"MAE: {metrics['MAE']:.2f} Pa")
    print(f"MAPE: {metrics['MAPE']:.2f}%")
    print(f"RMSE: {metrics['RMSE']:.2f} Pa")
    print(f"R2: {metrics['R2']:.4f}")


if __name__ == "__main__":
    main()
