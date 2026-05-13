import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class ExperimentLogger:
    def __init__(self):
        self.raw_data = []

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
        entry = {
            "Dataset": dataset,
            "Noise_Type": noise_type,
            "Intensity": intensity,
            "Avg_Accuracy": accuracy,
            "Classifier": classifier
        }
        if metric is not None:
            entry["Metric"] = metric
        self.raw_data.append(entry)

    def get_summary_table(self):
        """将结果转换为 pandas DataFrame 表格"""
        return pd.DataFrame(self.raw_data)

    def plot_results_comparison(self, df, title="Robustness Analysis"):
        """
        绘制对比图
        如果 dataframe 中包含多个分类器，会自动用不同颜色或线型区分
        """
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