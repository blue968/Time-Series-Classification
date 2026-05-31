import csv
from pathlib import Path

import numpy as np

from distances import DISTANCES_BY_CATEGORY
from ucr_dataset import dataset_domains


DISTANCE_CATEGORIES = {
    distance: category
    for category, distances in DISTANCES_BY_CATEGORY.items()
    for distance in distances
}


def get_dataset_category(dataset_name):
    return dataset_domains.get(str(dataset_name), "unknown_dataset_category")


def get_distance_category(distance_name):
    return DISTANCE_CATEGORIES.get(str(distance_name), "unknown_distance_category")


def get_statistics_dir(results_root, dataset_name, distance_name):
    return (
        Path(results_root)
        / get_dataset_category(dataset_name)
        / str(dataset_name)
        / get_distance_category(distance_name)
        / str(distance_name)
    )


def find_statistics_table(results_root, dataset_name, distance_name):
    table_path = get_statistics_dir(results_root, dataset_name, distance_name) / "accuracy.csv"
    if table_path.exists():
        return table_path

    legacy_path = Path(results_root) / str(dataset_name) / str(distance_name) / "accuracy.csv"
    if legacy_path.exists():
        return legacy_path

    return table_path

class ExperimentLogger:
    def __init__(self, results_root="results/statistcs"):
        self.raw_data = []
        self.results_root = Path(results_root)

    def _classifier_column(self, classifier):
        normalized = str(classifier).strip().lower().replace("-", "")
        aliases = {
            "1nn": "1nn",
            "1nearestneighbor": "1nn",
            "nearestneighbor": "1nn",
            "svm": "svm",
        }
        return aliases.get(normalized, normalized)

    def _noise_label(self, noise_type, intensity):
        return f"{noise_type}_{float(intensity):g}"

    def _result_dir(self, dataset, metric):
        distance_name = metric or "unknown_metric"
        return get_statistics_dir(self.results_root, dataset, distance_name)

    def _save_statistics_table(self, dataset, noise_type, intensity, accuracy, classifier, metric):
        result_dir = self._result_dir(dataset, metric)
        result_dir.mkdir(parents=True, exist_ok=True)
        table_path = result_dir / "accuracy.csv"

        noise_label = self._noise_label(noise_type, intensity)
        classifier_col = self._classifier_column(classifier)

        base_fields = ["Noise", "Noise_Type", "Intensity"]
        rows = []
        fieldnames = base_fields.copy()
        if table_path.exists():
            with table_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or base_fields)
                rows = list(reader)

        if classifier_col not in fieldnames:
            fieldnames.append(classifier_col)
        for field in base_fields:
            if field not in fieldnames:
                fieldnames.insert(base_fields.index(field), field)

        row = next((item for item in rows if item.get("Noise") == noise_label), None)
        if row is None:
            row = {
                "Noise": noise_label,
                "Noise_Type": str(noise_type),
                "Intensity": f"{float(intensity):g}",
            }
            rows.append(row)
        else:
            row["Noise_Type"] = str(noise_type)
            row["Intensity"] = f"{float(intensity):g}"

        row[classifier_col] = f"{float(accuracy):.6f}"
        rows.sort(key=lambda item: (item.get("Noise_Type", ""), float(item.get("Intensity", "nan"))))

        with table_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return table_path

    def _describe_array(self, arr, name=None, y=None):
        """打印数组的简要信息并尝试解释各维含义。"""
        a = np.asarray(arr)
        info = {
            'name': name or 'array',
            'shape': a.shape,
            'ndim': a.ndim,
            'dtype': str(a.dtype),
            'size': a.size
        }
        print(f"[{info['name']}] shape={info['shape']}, ndim={info['ndim']}, dtype={info['dtype']}, size={info['size']}")

        # 解释维度含义的启发式规则
        if a.ndim == 1:
            print(f"  Interpretation: 1D array -> [time_steps] (每个元素为时间点的值)，time_steps={a.shape[0]}")
        elif a.ndim == 2:
            n0, n1 = a.shape
            if y is not None and hasattr(y, '__len__') and len(y) == n0:
                print(f"  Interpretation: 2D array -> [n_instances, time_steps], n_instances={n0}, time_steps={n1}")
            elif n0 == 1:
                print(f"  Interpretation: 2D array -> [1, time_steps] (单个序列)，time_steps={n1}")
            else:
                print(f"  Interpretation: 2D array ambiguous -> [dim0={n0}, dim1={n1}] (可能是 [instances, time] 或 [channels, time])")
        elif a.ndim == 3:
            n0, n1, n2 = a.shape
            print(f"  Interpretation: 3D array -> [n_instances, n_channels, time_steps], n_instances={n0}, n_channels={n1}, time_steps={n2}")
        else:
            print(f"  Interpretation: {a.ndim}D array -> dimensions = {a.shape}")

        # 提供简单统计信息（作为一维数组时）或对每通道/样本的摘要
        if a.size > 0:
            flat = a.ravel()
            print(f"  Summary stats: min={flat.min():.4g}, max={flat.max():.4g}, mean={flat.mean():.4g}, std={flat.std():.4g}")
            print(f"  First values (flat, up to 10): {flat[:10].tolist()}")
        print()

    def _extract_series(self, X, idx, y=None):
        """从可能的形状中提取单条序列作为一维 numpy 数组。"""
        X = np.asarray(X)
        if X.ndim == 1:
            return X
        if X.ndim == 2:
            if y is not None and hasattr(y, '__len__') and X.shape[0] == len(y):
                return X[idx]
            elif X.shape[0] == 1:
                return X[0]
            else:
                return X[0]
        if X.ndim == 3:
            n_instances = X.shape[0]
            if idx >= n_instances:
                idx = 0
            series = X[idx]
            if series.ndim == 2:
                return series[0]
            return series
        return X.reshape(-1)

    def plot_sample_series(self, X, y, title="Time Series Sample", alpha=0.6, max_timesteps=500):
        """可视化并打印关于输入数组形状和统计的详细信息。

        X: array-like, 支持 1D/2D/3D（常见格式: (n_instances, n_channels, time)）
        y: labels, 用于从每个类别选取样本
        """
        import matplotlib.pyplot as plt

        self._describe_array(X, name='X', y=y)
        self._describe_array(y, name='y')

        plt.figure(figsize=(12, 5))
        unique_labels = np.unique(y)

        # 从每个类别中取一个样本进行绘制
        for label in unique_labels:
            idx = np.where(y == label)[0][0]
            series = self._extract_series(X, idx, y=y)

            # 如果序列太长，进行采样以减少绘制的 time steps
            if len(series) > max_timesteps:
                step = max(1, len(series) // max_timesteps)
                series = series[::step]

            plt.plot(series, label=f"Class {label}", alpha=alpha, linewidth=1.5)

        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def log_result(self, dataset, noise_type, intensity, accuracy, classifier="1-NN", metric=None):
        """记录实验结果。可选包含 metric 字段。"""
        classifier_col = self._classifier_column(classifier)
        entry = {
            "Dataset": dataset,
            "Noise_Type": noise_type,
            "Intensity": intensity,
            "Avg_Accuracy": accuracy,
            "Classifier": classifier_col
        }
        if metric is not None:
            entry["Metric"] = metric
        self.raw_data.append(entry)
        table_path = self._save_statistics_table(
            dataset=dataset,
            noise_type=noise_type,
            intensity=intensity,
            accuracy=accuracy,
            classifier=classifier_col,
            metric=metric,
        )
        print(f"Saved statistics table: {table_path}")

    def get_summary_table(self):
        """将结果转换为 pandas DataFrame 表格"""
        import pandas as pd

        return pd.DataFrame(self.raw_data)

    def plot_results_comparison(self, df, title="Robustness Analysis"):
        """
        绘制对比图
        如果 dataframe 中包含多个分类器，会自动用不同颜色或线型区分
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(11, 6))
        # 使用 hue 区分噪声类型，使用 style 区分分类器 (1-NN vs SVM)
        sns.lineplot(data=df, x="Intensity", y="Avg_Accuracy", 
                    hue="Noise_Type", style="Classifier", 
                    markers=True, markersize=8)

        plt.title(title)
        plt.ylabel("Mean Accuracy (5-Fold CV)")
        plt.xlabel("Noise Intensity")
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()


def plot_saved_results(dataset_name, distance_name, results_root="results/statistcs", output_root="results/plots", show=False):
    """
    Read a saved statistics table and draw classifier robustness comparisons.

    Expected input:
        results/statistcs/{dataset_category}/{dataset_name}/{distance_category}/{distance_name}/accuracy.csv
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    table_path = find_statistics_table(results_root, dataset_name, distance_name)
    if not table_path.exists():
        raise FileNotFoundError(f"Statistics table not found: {table_path}")

    df = pd.read_csv(table_path)
    classifier_cols = [col for col in df.columns if col not in {"Noise", "Noise_Type", "Intensity"}]
    if not classifier_cols:
        raise ValueError(f"No classifier columns found in {table_path}")

    long_df = df.melt(
        id_vars=["Noise", "Noise_Type", "Intensity"],
        value_vars=classifier_cols,
        var_name="Classifier",
        value_name="Accuracy",
    ).dropna(subset=["Accuracy"])
    long_df["Intensity"] = pd.to_numeric(long_df["Intensity"], errors="coerce")
    long_df["Accuracy"] = pd.to_numeric(long_df["Accuracy"], errors="coerce")
    long_df = long_df.dropna(subset=["Intensity", "Accuracy"])

    noise_types = list(long_df["Noise_Type"].dropna().unique())
    if not noise_types:
        raise ValueError(f"No plottable noise rows found in {table_path}")

    sns.set_theme(style="whitegrid")
    n_cols = min(3, len(noise_types))
    n_rows = int(np.ceil(len(noise_types) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows), squeeze=False)

    for ax, noise_type in zip(axes.flat, noise_types):
        subset = long_df[long_df["Noise_Type"] == noise_type].sort_values("Intensity")
        sns.lineplot(
            data=subset,
            x="Intensity",
            y="Accuracy",
            hue="Classifier",
            marker="o",
            linewidth=2,
            ax=ax,
        )
        ax.set_title(str(noise_type))
        ax.set_xlabel("Noise intensity")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(title="Classifier")

    for ax in axes.flat[len(noise_types):]:
        ax.axis("off")

    fig.suptitle(f"{dataset_name} / {distance_name}: classifier robustness", fontsize=14)
    fig.tight_layout()

    output_dir = (
        Path(output_root)
        / get_dataset_category(dataset_name)
        / str(dataset_name)
        / get_distance_category(distance_name)
        / str(distance_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "accuracy_comparison.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return figure_path


def plot_overall_results(results_root="results/statistcs", output_root="results/plots", show=False):
    """
    Aggregate all saved statistics tables and draw an overall robustness plot.

    Each curve is the mean accuracy over all available dataset/distance pairs
    for the same noise type, intensity, and classifier.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    results_root = Path(results_root)
    tables = list(results_root.glob("*/*/*/*/accuracy.csv"))
    tables.extend(results_root.glob("*/*/accuracy.csv"))
    if not tables:
        raise FileNotFoundError(f"No statistics tables found under {results_root}")

    frames = []
    for table_path in tables:
        if len(table_path.relative_to(results_root).parts) == 5:
            dataset_category, dataset_name, distance_category, distance_name, _ = table_path.relative_to(results_root).parts
        else:
            distance_name = table_path.parent.name
            dataset_name = table_path.parent.parent.name
            dataset_category = get_dataset_category(dataset_name)
            distance_category = get_distance_category(distance_name)
        df = pd.read_csv(table_path)
        classifier_cols = [col for col in df.columns if col not in {"Noise", "Noise_Type", "Intensity"}]
        if not classifier_cols:
            continue

        long_df = df.melt(
            id_vars=["Noise", "Noise_Type", "Intensity"],
            value_vars=classifier_cols,
            var_name="Classifier",
            value_name="Accuracy",
        ).dropna(subset=["Accuracy"])
        long_df["Dataset_Category"] = dataset_category
        long_df["Dataset"] = dataset_name
        long_df["Distance_Category"] = distance_category
        long_df["Distance"] = distance_name
        frames.append(long_df)

    if not frames:
        raise ValueError(f"No plottable classifier columns found under {results_root}")

    all_results = pd.concat(frames, ignore_index=True)
    all_results["Intensity"] = pd.to_numeric(all_results["Intensity"], errors="coerce")
    all_results["Accuracy"] = pd.to_numeric(all_results["Accuracy"], errors="coerce")
    all_results = all_results.dropna(subset=["Intensity", "Accuracy"])

    summary = (
        all_results
        .groupby(["Noise_Type", "Intensity", "Classifier"], as_index=False)
        .agg(Accuracy=("Accuracy", "mean"))
    )

    noise_types = list(summary["Noise_Type"].dropna().unique())
    if not noise_types:
        raise ValueError(f"No plottable noise rows found under {results_root}")

    sns.set_theme(style="whitegrid")
    n_cols = min(3, len(noise_types))
    n_rows = int(np.ceil(len(noise_types) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows), squeeze=False)

    for ax, noise_type in zip(axes.flat, noise_types):
        subset = summary[summary["Noise_Type"] == noise_type].sort_values("Intensity")
        sns.lineplot(
            data=subset,
            x="Intensity",
            y="Accuracy",
            hue="Classifier",
            marker="o",
            linewidth=2,
            ax=ax,
        )
        ax.set_title(str(noise_type))
        ax.set_xlabel("Noise intensity")
        ax.set_ylabel("Mean accuracy")
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(title="Classifier")

    for ax in axes.flat[len(noise_types):]:
        ax.axis("off")

    fig.suptitle("Overall classifier robustness", fontsize=14)
    fig.tight_layout()

    output_dir = Path(output_root) / "overall"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "accuracy_comparison.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return figure_path
