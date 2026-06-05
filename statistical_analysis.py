"""
Time Series Classification -- Statistical Analysis
==================================================
1. Aggregate all accuracy.csv files under results/statistcs/ into results/summary.csv
2. Fit mixed-effects model:
      accuracy ~ C(metric) * C(classifier) * C(noise_type) * C(intensity) + (1|dataset)
3. Report Wald tests for all main effects and interactions.

Usage:
    python statistical_analysis.py              # run both steps
    python statistical_analysis.py --model-only # skip aggregation, only fit model
    python statistical_analysis.py --agg-only   # only aggregate CSVs

Dependencies (install in project venv):
    pip install statsmodels pandas numpy
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Step 1: Aggregate individual accuracy.csv files -> summary table
# ============================================================================

def aggregate_results(stat_dir: str = "results/statistcs",
                      output_path: str = "results/summary.csv") -> pd.DataFrame:
    """Walk stat_dir, read every accuracy.csv, melt 1nn/svm -> classifier rows."""
    root = Path(stat_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root.resolve()}")

    csv_files = sorted(root.rglob("accuracy.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No accuracy.csv files found under {root}")

    print(f"[1] Found {len(csv_files)} accuracy.csv files -- aggregating...")

    rows = []
    for fp in csv_files:
        # Path: results/statistcs/{category}/{dataset}/{family}/{metric}/accuracy.csv
        parts = fp.parts
        metric = parts[-2]
        dataset = parts[-4]

        with open(fp, "r", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                noise_type = row["Noise_Type"].strip()
                intensity = row["Intensity"].strip()
                for clf in ("1nn", "svm"):
                    acc = row[clf].strip()
                    rows.append({
                        "dataset": dataset,
                        "metric": metric,
                        "classifier": clf,
                        "noise_type": noise_type,
                        "intensity": float(intensity),
                        "accuracy": float(acc),
                    })

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"    -> Saved {len(df)} rows to {Path(output_path).resolve()}")
    return df


# ============================================================================
# Step 2: Mixed-effects model & Wald tests
# ============================================================================

def fit_mixed_model(summary_path: str = "results/summary.csv",
                    output_dir: str = "results"):
    """Fit the mixed-effects model and report significance of all terms."""
    # ---- load data ----
    print(f"\n[2] Loading {summary_path} ...")
    df = pd.read_csv(summary_path)
    print(f"    {len(df)} rows, columns: {list(df.columns)}")

    # Ensure categorical columns are treated as such in the formula
    for col in ("metric", "classifier", "noise_type", "dataset"):
        df[col] = df[col].astype(str)

    # intensity -> categorical (experimental conditions, not continuous)
    df["intensity"] = df["intensity"].astype(str)

    # ---- formula ----
    # Full factorial: all main effects + all 2/3/4-way interactions
    formula = (
        "accuracy ~ C(metric) * C(classifier) * C(noise_type) * C(intensity)"
    )
    print(f"\n    Formula: {formula}")
    print(f"    Random effect: (1 | dataset)")

    # ---- fit ----
    print("\n    Fitting mixed-effects model (REML) ...")
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM
    except ImportError:
        print("\nERROR: statsmodels is required. Install it with:")
        print("    pip install statsmodels")
        sys.exit(1)

    model = MixedLM.from_formula(
        formula,
        data=df,
        groups="dataset",
        re_formula="1",
    )

    # Try LBFGS first; fall back to Powell if it fails
    try:
        result = model.fit(method=["lbfgs", "powell"], maxiter=500)
    except Exception:
        print("    LBFGS/Powell failed, trying BFGS ...")
        result = model.fit(method="bfgs", maxiter=500)

    print(f"    Converged: {result.converged}")
    print(f"    Log-likelihood: {result.llf:.4f}")
    print(f"    N obs: {result.nobs}, N groups: {result.model.n_groups}")

    # ---- variance components ----
    vc = result.cov_re.iloc[0, 0] if hasattr(result, "cov_re") else result.cov_re
    if isinstance(vc, pd.DataFrame):
        vc = vc.iloc[0, 0]
    sigma_e = result.scale
    sigma_b = vc
    icc = sigma_b / (sigma_b + sigma_e) if (sigma_b + sigma_e) > 0 else 0
    print(f"\n    Variance components:")
    print(f"      var(dataset)  = {sigma_b:.6f}")
    print(f"      var(residual) = {sigma_e:.6f}")
    print(f"      ICC           = {icc:.4f}  (proportion of variance due to dataset)")

    # ---- Wald tests for fixed-effect terms (joint tests per term) ----
    print("\n" + "=" * 80)
    print("WALD TESTS FOR FIXED EFFECTS (joint term-level tests)")
    print("=" * 80)
    wald_records = _term_level_wald_table(result)

    # ---- model summary ----
    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    print(result.summary().tables[0])  # model info

    # ---- save results ----
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save Wald test table
    wald_path = out_dir / "mixed_model_wald_tests.csv"
    pd.DataFrame(wald_records).to_csv(wald_path, index=False)
    print(f"\n    Wald tests saved to {wald_path.resolve()}")

    # Save full coefficient table
    coef_path = out_dir / "mixed_model_coefficients.csv"
    # Save coefficients: statsmodels summary().tables[1] may be a DataFrame or SimpleTable
    coef_tbl = result.summary().tables[1]
    if isinstance(coef_tbl, pd.DataFrame):
        coef_tbl.to_csv(coef_path)
    else:
        with open(coef_path, "w", encoding="utf-8") as fh:
            fh.write(str(coef_tbl))
    print(f"    Coefficients saved to {coef_path.resolve()}")

    # ---- model diagnostics ----
    print("\n" + "=" * 80)
    print("MODEL DIAGNOSTICS")
    print("=" * 80)
    diagnose_model(result, df, output_dir)

    # ---- simple effects ----
    print("\n" + "=" * 80)
    print("SIMPLE EFFECTS ANALYSIS")
    print("=" * 80)
    simple_effects_analysis(result, df, output_dir)

    print("\nDone.")
    return result


# ============================================================================
# Step 3: Model diagnostics
# ============================================================================

def diagnose_model(result, df: pd.DataFrame, output_dir: str = "results"):
    """Generate residual diagnostic plots and numerical checks for the LMM."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import shapiro, levene, norm

    plot_dir = Path(output_dir) / "plots" / "diagnostics"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ---- extract residuals ----
    # NOTE: In statsmodels MixedLM, result.resid = y - Xβ - Zb (CONDITIONAL residuals).
    # We add the BLUPs back to get marginal (population-level) residuals.
    sigma_e = np.sqrt(result.scale)
    vc = result.cov_re.iloc[0, 0] if hasattr(result, "cov_re") else result.cov_re
    if isinstance(vc, pd.DataFrame):
        vc = vc.iloc[0, 0]
    sigma_b = vc
    icc = sigma_b / (sigma_b + sigma_e**2) if (sigma_b + sigma_e**2) > 0 else 0

    # Conditional residuals & fitted (from model)
    resid_cond = result.resid.values
    fitted_cond = result.fittedvalues.values

    # Random effects (BLUPs) mapped to each observation
    re_raw = result.random_effects
    re_dict = {}
    for k, v in re_raw.items():
        if isinstance(v, (pd.Series, pd.DataFrame)):
            re_dict[str(k)] = float(v.iloc[0])
        elif isinstance(v, np.ndarray):
            re_dict[str(k)] = float(v.flat[0])
        else:
            re_dict[str(k)] = float(v)
    re_values = np.array([re_dict.get(str(g), 0.0) for g in df["dataset"]])

    # Marginal residuals: add BLUPs back to conditional residuals
    resid_marg = resid_cond + re_values
    fitted_marg = fitted_cond - re_values

    std_resid_cond = resid_cond / sigma_e

    # ---- numerical diagnostics ----
    print(f"\n    Residual diagnostics (conditional residuals):")
    print(f"      Mean  = {np.mean(resid_cond):.6f}")
    print(f"      SD    = {np.std(resid_cond):.6f}  (model sigma_e = {sigma_e:.6f})")
    skew = float(pd.Series(resid_cond).skew())
    kurt = float(pd.Series(resid_cond).kurtosis())
    print(f"      Skewness = {skew:.4f}")
    print(f"      Kurtosis = {kurt:.4f}  (excess; 0 = normal)")

    # Shapiro-Wilk (sample max 5000)
    n_for_test = min(len(resid_cond), 5000)
    rng = np.random.default_rng(42)
    idx_sample = rng.choice(len(resid_cond), n_for_test, replace=False)
    sw_stat, sw_p = shapiro(resid_cond[idx_sample])
    print(f"      Shapiro-Wilk (n={n_for_test}): W={sw_stat:.4f}, p={sw_p:.6f}")

    # Levene test for homoscedasticity across datasets
    groups = df["dataset"].values
    unique_groups = sorted(set(groups))
    group_resids = [resid_cond[groups == g] for g in unique_groups]
    lev_stat, lev_p = levene(*group_resids)
    print(f"      Levene test (across datasets): stat={lev_stat:.4f}, p={lev_p:.6f}")

    # Conditional R2: 1 - var(cond_resid) / var(y)
    r2_cond = 1.0 - np.var(resid_cond) / np.var(df["accuracy"].values)
    # Marginal R2: 1 - var(marg_resid) / var(y)
    r2_marg = 1.0 - np.var(resid_marg) / np.var(df["accuracy"].values)
    print(f"      Marginal R2  = {r2_marg:.4f}")
    print(f"      Conditional R2 = {r2_cond:.4f}")

    # ---- generate diagnostic plots ----
    print(f"\n    Generating diagnostic plots -> {plot_dir.resolve()} ...")

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    fig.suptitle("LMM Diagnostic Plots  |  "
                 f"Marginal R2={r2_marg:.3f}  Conditional R2={r2_cond:.3f}  ICC={icc:.4f}",
                 fontsize=13, fontweight="bold", y=0.98)

    # (1) Residuals vs Fitted (conditional)
    ax = axes[0, 0]
    ax.scatter(fitted_cond, resid_cond, alpha=0.3, s=8, edgecolors="none", c="steelblue")
    ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8)
    try:
        from numpy.polynomial.polynomial import polyfit
        order = 2
        x_sorted = np.sort(fitted_cond)
        coeffs = polyfit(fitted_cond, resid_cond, order)
        y_smooth = sum(c * x_sorted ** i for i, c in enumerate(coeffs))
        ax.plot(x_sorted, y_smooth, color="darkorange", linewidth=2, label="quadratic trend")
        ax.legend(fontsize=8)
    except Exception:
        pass
    ax.set_xlabel("Fitted (conditional)")
    ax.set_ylabel("Conditional Residual")
    ax.set_title("Residuals vs Fitted")

    # (2) QQ plot
    ax = axes[0, 1]
    sorted_resid = np.sort(std_resid_cond)
    theoretical = norm.ppf((np.arange(len(sorted_resid)) + 0.5) / len(sorted_resid))
    ax.scatter(theoretical, sorted_resid, alpha=0.3, s=8, c="steelblue")
    ax.plot([-4, 4], [-4, 4], "r--", linewidth=0.8)
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel("Theoretical N(0,1) quantiles")
    ax.set_ylabel("Standardized Conditional Residuals")
    ax.set_title(f"QQ Plot  (Shapiro-Wilk p={sw_p:.4f})")

    # (3) Histogram + KDE
    ax = axes[0, 2]
    ax.hist(std_resid_cond, bins=60, density=True, alpha=0.6, color="steelblue", edgecolor="white")
    x_grid = np.linspace(-4, 4, 200)
    ax.plot(x_grid, norm.pdf(x_grid), "r-", linewidth=2, label="N(0,1)")
    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(std_resid_cond)
    ax.plot(x_grid, kde(x_grid), "darkorange", linewidth=2, label="KDE")
    ax.set_xlim(-4, 4)
    ax.set_xlabel("Standardized Conditional Residual")
    ax.set_ylabel("Density")
    ax.set_title(f"Residual Distribution  (skew={skew:.3f}, kurt={kurt:.3f})")
    ax.legend(fontsize=8)

    # (4) Scale-Location
    ax = axes[1, 0]
    sqrt_abs_resid = np.sqrt(np.abs(std_resid_cond))
    ax.scatter(fitted_cond, sqrt_abs_resid, alpha=0.3, s=8, c="steelblue")
    try:
        x_sorted_sl = np.sort(fitted_cond)
        coeffs = polyfit(fitted_cond, sqrt_abs_resid, 2)
        y_smooth_sl = sum(c * x_sorted_sl ** i for i, c in enumerate(coeffs))
        ax.plot(x_sorted_sl, y_smooth_sl, color="darkorange", linewidth=2)
    except Exception:
        pass
    ax.set_xlabel("Fitted (conditional)")
    ax.set_ylabel("sqrt(|Std. Conditional Residual|)")
    ax.set_title("Scale-Location")

    # (5) Residuals by dataset (BLUPs + conditional)
    ax = axes[1, 1]
    # Show BLUP (random intercept) per dataset
    ds_names = sorted(unique_groups, key=lambda g: re_dict.get(str(g), 0.0))
    blups = [re_dict.get(str(g), 0.0) for g in ds_names]
    colors = ["#2c7bb6" if v >= 0 else "#d7191c" for v in blups]
    ax.barh(range(len(ds_names)), blups, color=colors, alpha=0.8, height=0.7)
    ax.set_yticks(range(len(ds_names)))
    ax.set_yticklabels(ds_names, fontsize=7)
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_xlabel("BLUP (Random Intercept)")
    ax.set_title("Random Effects by Dataset (BLUPs)")

    # (6) Residuals by noise_type × intensity
    ax = axes[1, 2]
    df_plot = df.copy()
    df_plot["cond_resid"] = resid_cond
    order = [f"{nt}_{iv}" for nt in ["jittering", "missing", "outlier"]
             for iv in sorted(df["intensity"].unique())]
    palette = {nt: c for nt, c in zip(["jittering", "missing", "outlier"],
                                        ["#2c7bb6", "#abd9e9", "#d7191c"])}
    df_plot["noise_intensity"] = df_plot["noise_type"] + "_" + df_plot["intensity"]
    sns.boxplot(data=df_plot, x="noise_intensity", y="cond_resid",
                hue="noise_type", palette=palette, order=order,
                ax=ax, legend=False, fliersize=2, linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("")
    ax.set_xticklabels([o.replace("_", "\n") for o in order], rotation=45, fontsize=6, ha="right")
    ax.set_ylabel("Conditional Residual")
    ax.set_title("Residuals by Noise Type × Intensity")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = plot_dir / "lmm_diagnostics.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {fig_path.resolve()}")

    # ---- additional: residual autocorrelation check (by dataset, sorted) ----
    # For each dataset, check if there's within-dataset residual correlation
    acf_summary = []
    for g in unique_groups:
        mask = groups == g
        g_resid = resid_cond[mask]
        if len(g_resid) > 2:
            # Compute lag-1 autocorrelation
            acf1 = np.corrcoef(g_resid[:-1], g_resid[1:])[0, 1]
            if not np.isnan(acf1):
                acf_summary.append(acf1)
    if acf_summary:
        acf_mean = np.mean(acf_summary)
        acf_max = np.max(np.abs(acf_summary))
        print(f"      Within-dataset lag-1 ACF: mean={acf_mean:.4f}, max|ACF|={acf_max:.4f}")

    print("    Diagnostics complete.")


# ============================================================================
# Step 4: Simple effects analysis — decompose significant interactions
# ============================================================================

def simple_effects_analysis(result, df: pd.DataFrame, output_dir: str = "results"):
    """Decompose metric × intensity and metric × noise_type × intensity interactions.

    Produces:
      1. For each noise_type × intensity panel: pairwise metric comparisons
         with FDR-corrected p-values, CIs, and estimated differences.
      2. For each metric × noise_type panel: intensity trend (linear contrast).

    All estimates are marginal predictions from the fitted mixed model,
    computed via the delta method using the coefficient covariance matrix.
    """
    from patsy import build_design_matrices
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- setup ----
    design_info = result.model.data.design_info
    # params include 'dataset Var' at the end — use only fixed-effect portion
    n_fixed = len(result.model.exog_names)  # 450
    beta = result.params.values[:n_fixed]
    cov_full = result.cov_params().values
    cov_beta = cov_full[:n_fixed, :n_fixed]  # fixed-effects covariance only

    metrics = sorted(df["metric"].unique())
    noise_types = sorted(df["noise_type"].unique())
    intensities = sorted(df["intensity"].unique())
    ref_classifier = "1nn"

    # ---- build reference grid ----
    grid_rows = []
    for m in metrics:
        for nt in noise_types:
            for iv in intensities:
                grid_rows.append({
                    "metric": m,
                    "classifier": ref_classifier,
                    "noise_type": nt,
                    "intensity": iv,
                })
    grid_df = pd.DataFrame(grid_rows)

    # Build design matrix for reference grid (column order matches model's β)
    (X_grid,) = build_design_matrices([design_info], grid_df, return_type="matrix")
    X_grid = np.asarray(X_grid)

    # Predicted accuracy at each grid point
    pred = X_grid @ beta

    n_grid = len(grid_df)
    print(f"\n    Reference grid: {n_grid} points ({len(metrics)} metrics x "
          f"{len(noise_types)} noise_types x {len(intensities)} intensities)")

    # ---- helper: contrast test ----
    def contrast_ci(c_vec):
        """Return (estimate, SE, z, p, ci_low, ci_high) for contrast vector c_vec."""
        est = float(c_vec @ beta)
        se = float(np.sqrt(c_vec @ cov_beta @ c_vec))
        z = est / se if se > 1e-12 else 0.0
        p = float(2.0 * norm.sf(abs(z)))
        ci_lo = est - 1.96 * se
        ci_hi = est + 1.96 * se
        return est, se, z, p, ci_lo, ci_hi

    # ========================================================================
    # Part A: metric pairwise comparisons within each noise_type x intensity
    # ========================================================================
    print("\n    --- Part A: metric simple effects within each noise_type x intensity ---")

    all_comparisons = []
    for nt in noise_types:
        for iv in intensities:
            mask = (grid_df["noise_type"] == nt) & (grid_df["intensity"] == iv)
            idxs = np.where(mask)[0]
            metric_at_idx = {grid_df.iloc[i]["metric"]: i for i in idxs}

            for i_m1 in range(len(metrics)):
                for i_m2 in range(i_m1 + 1, len(metrics)):
                    m1, m2 = metrics[i_m1], metrics[i_m2]
                    if m1 not in metric_at_idx or m2 not in metric_at_idx:
                        continue
                    i1, i2 = metric_at_idx[m1], metric_at_idx[m2]
                    c_vec = np.zeros(len(beta))
                    c_vec[i1] = 1.0
                    c_vec[i2] = -1.0
                    # The difference is row[i1] - row[i2] = (X[i1]-X[i2]) @ β
                    # Build the contrast directly:
                    c_vec_full = X_grid[i1] - X_grid[i2]
                    est, se, z, p_raw, ci_lo, ci_hi = contrast_ci(c_vec_full)
                    all_comparisons.append({
                        "noise_type": nt, "intensity": iv,
                        "metric_A": m1, "metric_B": m2,
                        "diff": round(est, 6), "SE": round(se, 6),
                        "z": round(z, 4), "p_raw": round(p_raw, 6),
                        "CI_low": round(ci_lo, 6), "CI_high": round(ci_hi, 6),
                        "pred_A": round(pred[i1], 6),
                        "pred_B": round(pred[i2], 6),
                    })

    comp_df = pd.DataFrame(all_comparisons)
    # FDR correction across all comparisons
    _, p_corrected, _, _ = multipletests(comp_df["p_raw"].values, method="fdr_bh")
    comp_df["p_fdr"] = p_corrected

    # Summary statistics
    n_total = len(comp_df)
    n_sig_05 = (comp_df["p_raw"] < 0.05).sum()
    n_sig_fdr = (comp_df["p_fdr"] < 0.05).sum()
    print(f"      Total pairwise metric comparisons: {n_total}")
    print(f"      Significant at raw p<0.05: {n_sig_05} ({100*n_sig_05/n_total:.1f}%)")
    print(f"      Significant at FDR<0.05:  {n_sig_fdr} ({100*n_sig_fdr/n_total:.1f}%)")

    # ---- Part A summary: best metric(s) per panel ----
    print("\n    Best-performing metrics per noise_type x intensity panel:")
    print(f"      {'Panel':<30s} {'Best Metric(s)':<25s} {'Accuracy':>8s}  {'vs worst':>10s}")
    print(f"      {'-'*30} {'-'*25} {'-'*8}  {'-'*10}")

    panel_bests = []
    for nt in noise_types:
        for iv in intensities:
            mask = (grid_df["noise_type"] == nt) & (grid_df["intensity"] == iv)
            idxs = np.where(mask)[0]
            best_idx = idxs[pred[idxs].argmax()]
            worst_idx = idxs[pred[idxs].argmin()]
            best_m = grid_df.iloc[best_idx]["metric"]
            best_acc = pred[best_idx]
            worst_acc = pred[worst_idx]
            margin = best_acc - worst_acc
            # Find all metrics not significantly different from best (FDR)
            panel_mask = (comp_df["noise_type"] == nt) & (comp_df["intensity"] == iv)
            panel_comps = comp_df[panel_mask]
            tied = set()
            for _, row in panel_comps.iterrows():
                if row["p_fdr"] >= 0.05:
                    if row["metric_A"] == best_m:
                        tied.add(row["metric_B"])
                    elif row["metric_B"] == best_m:
                        tied.add(row["metric_A"])
            tied_str = ", ".join(sorted(tied)) if tied else best_m
            panel_label = f"{nt}/{iv}"
            print(f"      {panel_label:<30s} {best_m:<25s} {best_acc:>8.4f}  {margin:>+10.6f}")
            if len(tied) > 1:
                print(f"      {'':30s} (tied: {tied_str})")
            panel_bests.append({
                "noise_type": nt, "intensity": iv,
                "best_metric": best_m, "best_acc": round(best_acc, 6),
                "worst_acc": round(worst_acc, 6), "margin": round(margin, 6),
                "tied_metrics": tied_str,
            })

    # Save Part A
    comp_path = out_dir / "simple_effects_metric_pairs.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"\n    Pairwise comparisons saved to {comp_path.resolve()}")

    panel_path = out_dir / "simple_effects_best_metrics.csv"
    pd.DataFrame(panel_bests).to_csv(panel_path, index=False)
    print(f"    Best metrics per panel saved to {panel_path.resolve()}")

    # ========================================================================
    # Part B: intensity trends within each metric x noise_type
    # ========================================================================
    print("\n\n    --- Part B: intensity trends within each metric x noise_type ---")

    # For each metric × noise_type, compute the effect of each intensity level
    # relative to intensity=0 (baseline), and a linear trend contrast.
    iv_baseline = "0.0"
    trend_results = []

    # For each metric x noise_type, compute intensity effects vs baseline and linear trend
    for m in metrics:
        for nt in noise_types:
            mask = (grid_df["metric"] == m) & (grid_df["noise_type"] == nt)
            idxs = np.where(mask)[0]
            iv_at_idx = {grid_df.iloc[i]["intensity"]: i for i in idxs}

            # Find baseline (intensity=0)
            if iv_baseline not in iv_at_idx:
                continue
            idx_base = iv_at_idx[iv_baseline]

            # a) Each intensity vs baseline
            for iv in intensities:
                if iv == iv_baseline or iv not in iv_at_idx:
                    continue
                idx_iv = iv_at_idx[iv]
                c_vec = X_grid[idx_iv] - X_grid[idx_base]
                est, se, z, p_raw, ci_lo, ci_hi = contrast_ci(c_vec)
                trend_results.append({
                    "metric": m, "noise_type": nt,
                    "contrast": f"I{iv}-I{iv_baseline}",
                    "diff": round(est, 6), "SE": round(se, 6),
                    "z": round(z, 4), "p_raw": round(p_raw, 6),
                    "CI_low": round(ci_lo, 6), "CI_high": round(ci_hi, 6),
                })

            # b) Linear trend contrast: slope of accuracy vs intensity
            # slope = cov(iv, μ) / var(iv), computed via delta method
            iv_vals = np.array([float(iv) for iv in intensities])
            slope_den = np.sum((iv_vals - iv_vals.mean()) ** 2)
            c_slope = np.zeros(len(beta))
            for iv in intensities:
                if iv in iv_at_idx:
                    c_slope += ((float(iv) - iv_vals.mean()) / slope_den) * X_grid[iv_at_idx[iv]]
            est_slope, se_slope, z_slope, p_slope, ci_lo_s, ci_hi_s = contrast_ci(c_slope)
            trend_results.append({
                "metric": m, "noise_type": nt,
                "contrast": "linear_trend",
                "diff": round(est_slope, 6),
                "SE": round(se_slope, 6),
                "z": round(z_slope, 4),
                "p_raw": round(p_slope, 6),
                "CI_low": round(ci_lo_s, 6),
                "CI_high": round(ci_hi_s, 6),
            })

    trend_df = pd.DataFrame(trend_results)

    # FDR correction within contrast type (linear or pairwise)
    linear_mask = trend_df["contrast"] == "linear_trend"
    pairwise_mask = ~linear_mask
    for mask_type in [linear_mask, pairwise_mask]:
        _, p_corr, _, _ = multipletests(trend_df.loc[mask_type, "p_raw"].values, method="fdr_bh")
        trend_df.loc[mask_type, "p_fdr"] = p_corr
    trend_df["p_fdr"] = trend_df["p_fdr"].fillna(1.0)

    n_lin = linear_mask.sum()
    n_lin_sig = (trend_df.loc[linear_mask, "p_fdr"] < 0.05).sum()
    print(f"      Linear trend contrasts: {n_lin} ({n_lin_sig} significant at FDR<0.05)")
    print(f"      Pairwise vs baseline: {pairwise_mask.sum()}")

    # ---- Part B summary: metrics with strongest intensity degradation ----
    print(f"\n    Top 10 metrics by linear intensity degradation (per 0.1 intensity):")
    linear_trends = trend_df[linear_mask].sort_values("diff").head(10)
    for _, row in linear_trends.iterrows():
        sig = "*" if row["p_fdr"] < 0.05 else ""
        print(f"      {row['metric']:<15s} x {row['noise_type']:<10s}: "
              f"slope={row['diff']:+.6f}  SE={row['SE']:.6f}  "
              f"CI=[{row['CI_low']:+.6f}, {row['CI_high']:+.6f}]  {sig}")

    print(f"\n    Top 10 most robust metrics (least degradation):")
    robust = trend_df[linear_mask].sort_values("diff", ascending=False).head(10)
    for _, row in robust.iterrows():
        sig = "*" if row["p_fdr"] < 0.05 else ""
        print(f"      {row['metric']:<15s} x {row['noise_type']:<10s}: "
              f"slope={row['diff']:+.6f}  SE={row['SE']:.6f}  "
              f"CI=[{row['CI_low']:+.6f}, {row['CI_high']:+.6f}]  {sig}")

    # ---- intensity pairwise (largest drops from baseline) ----
    print(f"\n    Largest intensity-induced accuracy drops (intensity vs baseline):")
    pairwise = trend_df[pairwise_mask].sort_values("diff").head(15)
    for _, row in pairwise.iterrows():
        sig = "*" if row["p_fdr"] < 0.05 else ""
        print(f"      {row['metric']:<15s} x {row['noise_type']:<10s} {row['contrast']:<12s}: "
              f"diff={row['diff']:+.6f}  SE={row['SE']:.6f}  "
              f"CI=[{row['CI_low']:+.6f}, {row['CI_high']:+.6f}]  {sig}")

    # Save Part B
    trend_path = out_dir / "simple_effects_intensity_trends.csv"
    trend_df.to_csv(trend_path, index=False)
    print(f"\n    Intensity trend results saved to {trend_path.resolve()}")

    print("\n    Simple effects analysis complete.")
    return comp_df, trend_df, pd.DataFrame(panel_bests)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sig_stars(p: float) -> str:
    """Convert p-value to significance stars."""
    if np.isnan(p):
        return "?"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return ""


def _term_level_wald_table(result):
    """Compute joint Wald tests for each formula term by grouping parameters.

    Parses parameter names like ``C(metric)[T.ddtw]:C(classifier)[T.svm]``
    back into their parent term (e.g. ``C(metric):C(classifier)``) and tests
    the joint null that all coefficients for that term are zero.

    Returns a list of dicts for later CSV export.
    """
    import re
    from scipy.stats import chi2

    cov = result.cov_params()
    beta = result.params

    # ---- group parameters by term ----
    def _extract_term(name: str) -> str:
        """Strip level brackets to recover the formula term."""
        if name == "Intercept":
            return "Intercept"
        # Split on ':' then strip [T.xxx] from each component, rejoin
        parts = []
        for p in name.split(":"):
            p = re.sub(r"\[T\.[^\]]*\]", "", p.strip())
            parts.append(p)
        return ":".join(parts)

    term_params: dict[str, list[str]] = {}
    term_order: list[str] = []
    for pname in beta.index:
        term = _extract_term(pname)
        if term not in term_params:
            term_params[term] = []
            term_order.append(term)
        term_params[term].append(pname)

    # ---- compute Wald statistic for each term ----
    print(f"{'Term':<50s} {'df':>4s}  {'chi2':>10s}  {'p-value':>10s}  {'Sig.':>6s}")
    print("-" * 85)

    records = []
    for term in term_order:
        pnames = term_params[term]
        idxs = [beta.index.get_loc(p) for p in pnames]
        b_sub = beta.iloc[idxs].values
        cov_sub = cov.iloc[idxs, idxs].values

        # Use pseudo-inverse to handle potentially singular covariance
        try:
            cov_inv = np.linalg.inv(cov_sub)
            df_w = len(idxs)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov_sub)
            df_w = int(np.linalg.matrix_rank(cov_sub))

        w = float(b_sub.T @ cov_inv @ b_sub)
        p_val = float(1.0 - chi2.cdf(w, df_w)) if df_w > 0 else np.nan
        sig = _sig_stars(p_val)
        print(f"{term:<50s} {df_w:>4d}  {w:>10.4f}  {p_val:>10.6f}  {sig:>6s}")
        records.append({"term": term, "df": df_w, "chi2": round(w, 4), "p_value": round(p_val, 6), "significant": sig.strip()})

    return records


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Time Series Classification -- Statistical Analysis")
    parser.add_argument("--model-only", action="store_true",
                        help="Skip aggregation, only fit the mixed model")
    parser.add_argument("--agg-only", action="store_true",
                        help="Only aggregate CSVs, skip model fitting")
    args = parser.parse_args()

    need_model = not args.agg_only

    if not args.model_only:
        summary_path = "results/summary.csv"
        aggregate_results(output_path=summary_path)
    else:
        summary_path = "results/summary.csv"
        if not Path(summary_path).exists():
            print(f"summary.csv not found at {summary_path}. Run without --model-only first.")
            sys.exit(1)

    if need_model and not args.agg_only:
        fit_mixed_model(summary_path=summary_path)


if __name__ == "__main__":
    main()
