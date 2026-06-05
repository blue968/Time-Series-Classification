"""
Generate publication-quality figures from LMM results.

Figure 1: Interaction line plots
    - x-axis: intensity, y-axis: accuracy
    - Each line = one distance metric
    - Faceted by noise_type (columns), grouped by classifier (rows)
    - Marginal predictions from the fitted mixed model.

Figure 2: Simple effects gap chart
    - Heatmap + bar chart of best−worst metric accuracy gap
    - Per noise_type × intensity combination, split by classifier.

Usage:
    python plot_results.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.regression.mixed_linear_model import MixedLM
from patsy import build_design_matrices


# ============================================================================
# Setup
# ============================================================================

OUTPUT_DIR = Path("results/plots")
SUMMARY_PATH = Path("results/summary.csv")

# Color palette — 15 distinguishable colors for 15 metrics
METRIC_COLORS = sns.color_palette("tab20", 15)
# We'll map metrics to specific colors via a dict
METRIC_LIST = [
    "adtw", "ddtw", "dtw", "edr", "erp", "euclidean", "lcss",
    "manhattan", "minkowski", "msm", "sbd", "shape_dtw", "twe", "wddtw", "wdtw",
]
COLOR_DICT = dict(zip(METRIC_LIST, METRIC_COLORS))

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 150,
})


# ============================================================================
# Helper: build reference-grid predictions from the fitted model
# ============================================================================

def build_reference_predictions(summary_path: str = "results/summary.csv"):
    """Fit the mixed model and return a DataFrame of marginal predictions for
    all metric × classifier × noise_type × intensity combinations."""
    df = pd.read_csv(summary_path)
    for col in ("metric", "classifier", "noise_type", "dataset"):
        df[col] = df[col].astype(str)
    df["intensity"] = df["intensity"].astype(str)

    formula = "accuracy ~ C(metric) * C(classifier) * C(noise_type) * C(intensity)"
    model = MixedLM.from_formula(formula, data=df, groups="dataset", re_formula="1")
    result = model.fit(method=["lbfgs", "powell"], maxiter=500)
    print(f"Model fitted: converged={result.converged}, logLik={result.llf:.2f}")

    # Reference grid: all combinations
    metrics = sorted(df["metric"].unique())
    noise_types = sorted(df["noise_type"].unique())
    intensities = sorted(df["intensity"].unique(), key=float)
    classifiers = sorted(df["classifier"].unique())

    rows = []
    for m in metrics:
        for clf in classifiers:
            for nt in noise_types:
                for iv in intensities:
                    rows.append({"metric": m, "classifier": clf,
                                 "noise_type": nt, "intensity": iv})
    grid = pd.DataFrame(rows)

    # Design matrix
    design_info = result.model.data.design_info
    (X,) = build_design_matrices([design_info], grid, return_type="matrix")
    X = np.asarray(X)

    n_fixed = len(result.model.exog_names)
    beta = result.params.values[:n_fixed]
    pred = X @ beta

    grid["accuracy"] = pred
    grid["intensity_num"] = grid["intensity"].astype(float)
    return grid


# ============================================================================
# Figure 1: Interaction line plots
# ============================================================================

def fig1_interaction_lines(pred_df: pd.DataFrame):
    """Faceted line plot: intensity × accuracy, line per metric,
    columns = noise_type, rows = classifier."""
    classifiers = sorted(pred_df["classifier"].unique())
    noise_types = sorted(pred_df["noise_type"].unique())

    n_rows = len(classifiers)
    n_cols = len(noise_types)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows),
                             sharex=True, sharey=True, squeeze=False)

    for row_i, clf in enumerate(classifiers):
        clf_label = "1-NN" if clf == "1nn" else "SVM"
        for col_j, nt in enumerate(noise_types):
            ax = axes[row_i, col_j]
            sub = pred_df[(pred_df["classifier"] == clf) &
                          (pred_df["noise_type"] == nt)]

            for metric in sorted(pred_df["metric"].unique()):
                d = sub[sub["metric"] == metric].sort_values("intensity_num")
                ax.plot(d["intensity_num"], d["accuracy"],
                        marker="o", markersize=4, linewidth=1.5,
                        color=COLOR_DICT[metric], label=metric, alpha=0.85)

            ax.set_title(f"{nt.capitalize()}  |  {clf_label}", fontweight="bold")
            ax.set_xlabel("Intensity")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0.40, 1.00)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            # x-axis: use actual intensity values
            ax.set_xticks(sorted(sub["intensity_num"].unique()))

    # Single legend on the right
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", title="Metric",
               bbox_to_anchor=(1.12, 0.5), ncol=1, frameon=True,
               fontsize=7, title_fontsize=8)

    fig.suptitle("Interaction: Metric × Noise Type × Intensity",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 0.92, 0.98])

    path = OUTPUT_DIR / "interaction_lines.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 1 saved to {path.resolve()}")


# ============================================================================
# Figure 2: Best vs Worst metric gap — combined heatmap + bar chart
# ============================================================================

def fig2_gap_chart(pred_df: pd.DataFrame):
    """Show the max−min accuracy gap per noise_type × intensity × classifier.

    Left panel: heatmap (rows = noise_type × classifier, cols = intensity).
    Right panel: grouped bar chart for the largest-gap scenarios.
    """
    classifiers = sorted(pred_df["classifier"].unique())
    noise_types = sorted(pred_df["noise_type"].unique())
    intensities = sorted(pred_df["intensity"].unique(), key=float)

    # Compute gaps
    gap_records = []
    for clf in classifiers:
        for nt in noise_types:
            for iv in intensities:
                sub = pred_df[(pred_df["classifier"] == clf) &
                              (pred_df["noise_type"] == nt) &
                              (pred_df["intensity"] == iv)]
                if len(sub) == 0:
                    continue
                best = sub.loc[sub["accuracy"].idxmax()]
                worst = sub.loc[sub["accuracy"].idxmin()]
                gap_records.append({
                    "classifier": "1-NN" if clf == "1nn" else "SVM",
                    "noise_type": nt.capitalize(),
                    "intensity": iv,
                    "intensity_num": float(iv),
                    "best_metric": best["metric"],
                    "best_acc": round(best["accuracy"], 4),
                    "worst_metric": worst["metric"],
                    "worst_acc": round(worst["accuracy"], 4),
                    "gap": round(best["accuracy"] - worst["accuracy"], 4),
                })

    gap_df = pd.DataFrame(gap_records)
    gap_df["panel"] = gap_df["classifier"] + " | " + gap_df["noise_type"]

    # ---- 2A: Heatmap ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                             gridspec_kw={"width_ratios": [1, 1.3]})

    ax = axes[0]
    # Pivot: rows=panel, cols=intensity
    heat_data = gap_df.pivot_table(
        index="panel", columns="intensity_num", values="gap", aggfunc="first"
    )
    # Order panels: classifier first, then noise_type
    panel_order = sorted(heat_data.index, key=lambda x: (x.split("|")[0], x.split("|")[1]))
    heat_data = heat_data.loc[panel_order]

    sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, linecolor="white", ax=ax,
                cbar_kws={"label": "Best − Worst Accuracy Gap"},
                vmin=0, vmax=heat_data.values.max())
    ax.set_title("Best−Worst Metric Accuracy Gap\n(per Noise Type × Intensity × Classifier)",
                 fontweight="bold")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("")

    # ---- 2B: Grouped bar chart (largest gaps) ----
    ax = axes[1]

    # Select top 15 gaps for bar chart clarity
    top_gaps = gap_df.nlargest(15, "gap").sort_values("gap")
    y_labels = [
        f"{r['classifier']} | {r['noise_type']} I={r['intensity']}\n({r['best_metric']} vs {r['worst_metric']})"
        for _, r in top_gaps.iterrows()
    ]
    bars = ax.barh(y_labels, top_gaps["gap"].values, color="coral", edgecolor="white")

    # Annotate bars
    for bar, val in zip(bars, top_gaps["gap"].values):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    ax.set_xlabel("Accuracy Gap (Best − Worst)")
    ax.set_title("Largest Metric-Selection Gaps\n(Top 15 Scenarios)", fontweight="bold")
    ax.set_xlim(0, top_gaps["gap"].max() * 1.15)

    fig.suptitle("How Much Does Metric Choice Matter?",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    path = OUTPUT_DIR / "simple_effects_gap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 2 saved to {path.resolve()}")


# ============================================================================
# Figure 3: Robustness ranking — slope chart
# ============================================================================

def fig3_robustness_ranking(pred_df: pd.DataFrame):
    """Horizontal bar chart showing linear intensity degradation per metric × noise_type,
    split by classifier. Negative = accuracy drops with intensity. Red = sensitive, blue = robust.
    """
    classifiers = sorted(pred_df["classifier"].unique())
    noise_types = sorted(pred_df["noise_type"].unique())
    metric_list = sorted(pred_df["metric"].unique())

    fig, axes = plt.subplots(len(classifiers), len(noise_types),
                             figsize=(4.5 * len(noise_types), 5 * len(classifiers)),
                             sharex="col", squeeze=False)

    for row_i, clf in enumerate(classifiers):
        clf_label = "1-NN" if clf == "1nn" else "SVM"
        for col_j, nt in enumerate(noise_types):
            ax = axes[row_i, col_j]

            slopes = []
            for metric in metric_list:
                sub = pred_df[(pred_df["metric"] == metric) &
                              (pred_df["classifier"] == clf) &
                              (pred_df["noise_type"] == nt)]
                sub = sub.sort_values("intensity_num")
                iv_vals = sub["intensity_num"].values
                acc_vals = sub["accuracy"].values
                # Linear slope per unit intensity
                if len(iv_vals) >= 2 and np.std(iv_vals) > 0:
                    slope = np.polyfit(iv_vals, acc_vals, 1)[0]
                else:
                    slope = 0.0
                slopes.append({"metric": metric, "slope": slope})

            slope_df = pd.DataFrame(slopes).sort_values("slope")
            colors = ["#d7191c" if s < -1.0 else "#fdae61" if s < -0.5 else
                      "#2c7bb6" if s > -0.3 else "#abd9e9"
                      for s in slope_df["slope"].values]
            ax.barh(slope_df["metric"], slope_df["slope"], color=colors,
                    edgecolor="white", height=0.7)
            ax.axvline(x=0, color="black", linewidth=0.5, linestyle="--")
            ax.set_xlabel("Slope (accuracy / intensity unit)")
            ax.set_title(f"{nt.capitalize()}  |  {clf_label}", fontweight="bold")

    fig.suptitle("Metric Robustness: Accuracy Decline per Unit Intensity\n"
                 "(Red = highly sensitive, Blue = robust)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    path = OUTPUT_DIR / "robustness_slopes.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 3 saved to {path.resolve()}")


# ============================================================================
# Main
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building reference predictions from fitted model ...")
    pred_df = build_reference_predictions(SUMMARY_PATH)

    print("\nGenerating Figure 1: Interaction line plots ...")
    fig1_interaction_lines(pred_df)

    print("\nGenerating Figure 2: Best vs Worst gap chart ...")
    fig2_gap_chart(pred_df)

    print("\nGenerating Figure 3: Robustness ranking ...")
    fig3_robustness_ranking(pred_df)

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
