import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from distances import DISTANCES, DISTANCES_BY_CATEGORY


RESULTS_ROOT = "results/statistcs"
OUTPUT_ROOT = "results/plots"
NOISE_ORDER = ["jittering", "missing", "outlier"]
CLASSIFIER_ORDER = ["1nn", "svm"]
BASE_COLUMNS = {"Noise", "Noise_Type", "Intensity"}


DISTANCE_CATEGORY = {
    distance: category
    for category, distances in DISTANCES_BY_CATEGORY.items()
    for distance in distances
}


def _ordered(values, preferred):
    values = list(dict.fromkeys(values))
    preferred_values = [value for value in preferred if value in values]
    remaining = sorted(value for value in values if value not in preferred_values)
    return preferred_values + remaining


def _safe_name(value):
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _table_metadata(table_path, results_root):
    parts = table_path.relative_to(results_root).parts
    if len(parts) == 5:
        dataset_category, dataset, distance_category, distance, _ = parts
        return dataset_category, dataset, distance_category, distance
    if len(parts) == 3:
        dataset, distance, _ = parts
        return "unknown_dataset_category", dataset, DISTANCE_CATEGORY.get(distance, "unknown_distance_category"), distance
    raise ValueError(f"Unexpected statistics path: {table_path}")


def load_results(results_root=RESULTS_ROOT):
    results_root = Path(results_root)
    tables = list(results_root.glob("*/*/*/*/accuracy.csv"))
    tables.extend(results_root.glob("*/*/accuracy.csv"))
    if not tables:
        raise FileNotFoundError(f"No accuracy.csv files found under {results_root}")

    frames = []
    for table_path in tables:
        dataset_category, dataset, distance_category, distance = _table_metadata(table_path, results_root)
        df = pd.read_csv(table_path)
        classifier_cols = [col for col in df.columns if col not in BASE_COLUMNS]
        if not classifier_cols:
            continue

        long_df = df.melt(
            id_vars=["Noise", "Noise_Type", "Intensity"],
            value_vars=classifier_cols,
            var_name="Classifier",
            value_name="Accuracy",
        )
        long_df["Dataset_Category"] = dataset_category
        long_df["Dataset"] = dataset
        long_df["Distance_Category"] = distance_category
        long_df["Distance"] = distance
        frames.append(long_df)

    if not frames:
        raise ValueError(f"No classifier result columns found under {results_root}")

    data = pd.concat(frames, ignore_index=True)
    data["Intensity"] = pd.to_numeric(data["Intensity"], errors="coerce")
    data["Accuracy"] = pd.to_numeric(data["Accuracy"], errors="coerce")
    data = data.dropna(subset=["Noise_Type", "Intensity", "Accuracy"])
    data = data[data["Noise_Type"].isin(NOISE_ORDER)]
    data = data[data["Classifier"].notna()]
    return data


def _style_axes(ax):
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_distance_trends(data, output_root=OUTPUT_ROOT):
    """
    For each distance, draw one figure with one subplot per supported noise type.

    Each line is the mean accuracy over all available datasets at a given noise
    intensity, separated by classifier.
    """
    output_dir = Path(output_root) / "distance_trends"
    output_dir.mkdir(parents=True, exist_ok=True)

    noise_types = _ordered(data["Noise_Type"].unique(), NOISE_ORDER)
    distances = _ordered(data["Distance"].unique(), DISTANCES)
    classifiers = _ordered(data["Classifier"].unique(), CLASSIFIER_ORDER)
    colors = dict(zip(classifiers, plt.cm.Set2(np.linspace(0, 1, max(len(classifiers), 3)))))

    saved = []
    for distance in distances:
        distance_data = data[data["Distance"] == distance]
        if distance_data.empty:
            continue

        fig_width = max(9, 5.2 * len(noise_types))
        fig, axes = plt.subplots(1, len(noise_types), figsize=(fig_width, 4.8), sharey=True, squeeze=False)
        axes = axes.ravel()
        distance_category = distance_data["Distance_Category"].iloc[0]
        dataset_count = distance_data["Dataset"].nunique()

        for ax, noise_type in zip(axes, noise_types):
            subset = distance_data[distance_data["Noise_Type"] == noise_type]
            if subset.empty:
                ax.set_title(noise_type)
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axisbelow(True)
                _style_axes(ax)
                continue

            summary = (
                subset.groupby(["Intensity", "Classifier"], as_index=False)
                .agg(Accuracy=("Accuracy", "mean"))
                .sort_values("Intensity")
            )
            for classifier in classifiers:
                line_data = summary[summary["Classifier"] == classifier]
                if line_data.empty:
                    continue
                ax.plot(
                    line_data["Intensity"],
                    line_data["Accuracy"],
                    marker="o",
                    linewidth=2,
                    label=classifier,
                    color=colors[classifier],
                )

            ax.set_title(noise_type)
            ax.set_xlabel("Noise intensity")
            ax.set_ylim(0, 1.02)
            _style_axes(ax)

        axes[0].set_ylabel("Mean accuracy")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)

        fig.suptitle(
            f"{distance} ({distance_category}) accuracy under noise - mean over {dataset_count} datasets",
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0.08, 1, 0.9))

        figure_path = output_dir / f"{_safe_name(distance)}.png"
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(figure_path)

    return saved


def _add_relative_drop(data):
    baselines = data[data["Intensity"] == 0].copy()
    if baselines.empty:
        min_intensity = data.groupby(["Dataset", "Distance", "Classifier", "Noise_Type"])["Intensity"].transform("min")
        baselines = data[data["Intensity"].eq(min_intensity)].copy()

    baselines = baselines[
        ["Dataset", "Distance", "Classifier", "Noise_Type", "Accuracy"]
    ].rename(columns={"Accuracy": "Baseline_Accuracy"})
    baselines = baselines.drop_duplicates(["Dataset", "Distance", "Classifier", "Noise_Type"])

    merged = data.merge(
        baselines,
        on=["Dataset", "Distance", "Classifier", "Noise_Type"],
        how="left",
    )
    merged = merged.dropna(subset=["Baseline_Accuracy"])
    merged = merged[merged["Baseline_Accuracy"] != 0]
    merged["Relative_Drop"] = (
        (merged["Baseline_Accuracy"] - merged["Accuracy"]) / merged["Baseline_Accuracy"] * 100
    )
    return merged


def plot_relative_decline(data, output_root=OUTPUT_ROOT):
    """
    For each noise type, draw one figure with non-zero intensity subplots.

    Each subplot compares distances by their mean relative accuracy drop over
    all datasets. Relative drop is computed against intensity 0 for the same
    dataset, distance, classifier, and noise type.
    """
    output_dir = Path(output_root) / "relative_decline"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = _add_relative_drop(data)
    noise_types = _ordered(data["Noise_Type"].unique(), NOISE_ORDER)
    distances = _ordered(data["Distance"].unique(), DISTANCES)
    classifiers = _ordered(data["Classifier"].unique(), CLASSIFIER_ORDER)
    colors = dict(zip(classifiers, plt.cm.Set2(np.linspace(0, 1, max(len(classifiers), 3)))))

    saved = []
    for noise_type in noise_types:
        noise_data = data[data["Noise_Type"] == noise_type]
        if noise_data.empty:
            continue

        intensities = [value for value in sorted(noise_data["Intensity"].unique()) if value != 0][:5]
        if not intensities:
            continue
        fig_width = max(6, 4.8 * len(intensities))
        fig, axes = plt.subplots(1, len(intensities), figsize=(fig_width, 5), sharey=True, squeeze=False)
        axes = axes.ravel()

        for ax, intensity in zip(axes, intensities):
            subset = noise_data[noise_data["Intensity"] == intensity]
            summary = (
                subset.groupby(["Distance", "Classifier"], as_index=False)
                .agg(Relative_Drop=("Relative_Drop", "mean"))
            )

            x = np.arange(len(distances))
            width = 0.78 / max(len(classifiers), 1)
            for idx, classifier in enumerate(classifiers):
                cls_data = summary[summary["Classifier"] == classifier].set_index("Distance")
                values = [cls_data["Relative_Drop"].get(distance, np.nan) for distance in distances]
                offsets = x - 0.39 + width / 2 + idx * width
                ax.bar(
                    offsets,
                    values,
                    width=width,
                    label=classifier,
                    color=colors[classifier],
                    alpha=0.9,
                )

            ax.axhline(0, color="#444444", linewidth=0.8)
            ax.set_title(f"intensity={intensity:g}")
            ax.set_xticks(x)
            ax.set_xticklabels(distances, rotation=60, ha="right", fontsize=8)
            ax.set_xlabel("Distance")
            _style_axes(ax)

        axes[0].set_ylabel("Relative accuracy drop (%)")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)

        fig.suptitle(f"Relative performance decline under {noise_type}", fontsize=14)
        fig.tight_layout(rect=(0, 0.1, 1, 0.9))

        figure_path = output_dir / f"{_safe_name(noise_type)}.png"
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(figure_path)

    return saved


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations from saved experiment CSV files.")
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    args = parser.parse_args()

    data = load_results(args.results_root)
    distance_figures = plot_distance_trends(data, args.output_root)
    decline_figures = plot_relative_decline(data, args.output_root)

    print(f"Loaded {len(data)} rows from {args.results_root}")
    print(f"Saved {len(distance_figures)} distance trend figures to {Path(args.output_root) / 'distance_trends'}")
    print(f"Saved {len(decline_figures)} relative decline figures to {Path(args.output_root) / 'relative_decline'}")


if __name__ == "__main__":
    main()
