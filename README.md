# Time Series Classification

Robustness of Similarity Measures for Time Series Classification under Noise: An Empirical Study

## Overview

Time Series Classification (TSC) is widely used in domains such as sensors, ECG, images, motion signals, devices, speech, and other sequential data. In real-world scenarios, time series data often contain noise, missing values, sampling disturbance, outliers, or other deviations from the true signal.

This project studies how different similarity or distance measures behave under noisy conditions. The goal is to quantify and compare the performance degradation of time series classifiers when different noise types and intensities are introduced.

## Research Motivation

Many existing works propose new time series distance measures or improve existing measures with some discussion of robustness. However, these studies often compare only a small number of methods or treat noise robustness as a secondary property of a newly proposed method.

This project aims to provide a more systematic empirical comparison. Instead of covering every possible measure and every UCR dataset, the experiment selects representative datasets and representative similarity measures across categories. Noise is then injected into standard datasets to observe how classification accuracy changes.

The intended contribution is a practical comparison of robustness patterns across:

- Different dataset domains, such as image, sensor, motion, ECG, device, spectro, speech, traffic, and simulated data.
- Different similarity measure categories, such as lock-step, sliding, elastic, feature-based, edit-based, alignment, and embedding distances.
- Different noise types and intensities.
- Different classifiers, currently 1-NN and SVM.

## Related Work Context

Time series similarity measures form a broad research area, with many existing measures proposed for different assumptions and domains. Reviewing these measures gives a general view of how wide the design space is.

Some works focus specifically on improving TSC under noisy conditions by proposing robust distances or modifying existing ones. However, these methods are not always evaluated under a unified experimental setting.

Data augmentation and noise injection are also common techniques for studying robustness. In this project, noise is artificially added to standard UCR datasets so that different distance measures can be compared under controlled noisy environments.

## Methodology

### Classifiers

The project currently supports:

- `1nn`: one-nearest-neighbor classification using a selected time series distance.
- `svm`: support vector machine classification using a precomputed kernel derived from the distance matrix.
- `both`: run both classifiers.

### Dataset

The project uses the UCR Time Series Classification Archive 2018:

https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

The UCR archive contains 128 univariate time series datasets. Each dataset belongs to a specific domain or type, such as image, sensor, motion, ECG, device, spectro, traffic, and others.

The local dataset metadata is defined in `ucr_dataset.py`:

- `all_ucr_datasets`: ordered list of UCR datasets.
- `dataset_domains`: mapping from dataset name to dataset category.

The experiment does not need to cover every dataset in every run, but the framework is designed so that datasets from different domains can be selected for broader comparison.

### Similarity Measures

The distance measures are defined in `distances.py`. They are grouped by category:

- `lock_step`: Euclidean, Manhattan, Minkowski
- `sliding`: SBD
- `elastic`: DTW, WDTW, ADTW
- `feature_based`: ShapeDTW
- `edit_based`: TWE, MSM, ERP, EDR
- `alignment`: LCSS
- `embedding`: DDTW, WDDTW

The framework can run experiments over one or more selected measures. Results are saved according to both dataset category and distance category.

### Noise Types

Noise is treated as the deviation between the observed signal and the true signal. The current experiments focus on three implemented noise patterns:

1. Additive noise, such as Gaussian jittering.
2. Missing values, where parts of the sequence are removed or masked.
3. Outliers, where a small number of time points receive large deviations.

The current implementation supports the following noise names in `dataloader.py`:

- `jittering`
- `missing`
- `outlier`

## Evaluation

The primary metric is classification accuracy.

The main analysis focuses on:

- Accuracy under different noise types.
- Accuracy degradation as noise intensity increases.
- Differences between classifiers.
- Differences between distance categories.
- Differences between dataset domains.
- Runtime considerations, especially for expensive pairwise distances and SVM kernels.

## Project Structure

```text
.
├── main.py             # Experiment entry point
├── dataloader.py       # UCR loading, folds, and noise injection
├── distances.py        # Distance categories and distance/kernel utilities
├── knn.py              # 1-NN experiment
├── svm.py              # SVM experiment
├── logging_utils.py    # Result saving
├── plot.py             # Visualization from saved result tables
├── ucr_dataset.py      # UCR dataset names and domain metadata
├── requirements.txt    # Python dependencies
└── README.md
```

## Environment Setup

Create and activate a conda environment:

```bash
conda create -n ts_robustness python=3.10 -y
conda activate ts_robustness
```

Install core dependencies:

```bash
conda install -c conda-forge numpy pandas scikit-learn -y
```

Install the remaining project dependencies:

```bash
pip install -r requirements.txt
```

If you already have the local `tsc` environment, you can activate it directly:

```bash
conda activate tsc
```

## Running Experiments

Run the default experiment:

```bash
python main.py
```

Run only one classifier:

```bash
python main.py --experiment 1nn
python main.py --experiment svm
```

Run both classifiers:

```bash
python main.py --experiment both
```

Specify datasets and distance measures:

```bash
python main.py --datasets BeetleFly,Coffee --metrics euclidean,dtw,twe
```

## Plotting Existing Results

Visualization is separated from experiment execution. After experiments have generated CSV files under `results/statistcs/`, run:

```bash
python plot.py
```

This creates two groups of figures:

- `results/plots/distance_trends/`: one figure per distance. Each figure contains one subplot for each supported noise type. The curves show how the mean accuracy over all available datasets changes as noise intensity increases.
- `results/plots/relative_decline/`: one figure per noise type. Each figure contains one subplot for each non-zero noise intensity. The bars compare the relative accuracy drop of different distances, computed against the zero-noise baseline.

You can also choose custom input and output directories:

```bash
python plot.py --results-root results/statistcs --output-root results/plots
```

## Result Organization

Experiment statistics are saved as CSV files under:

```text
results/statistcs/{dataset_category}/{dataset_name}/{distance_category}/{distance_name}/accuracy.csv
```

For example:

```text
results/statistcs/spectro/Coffee/elastic/adtw/accuracy.csv
```

Each CSV table uses:

- Rows: different noise configurations.
- Columns: different classifiers, such as `1nn` and `svm`.

Plots are saved under:

```text
results/plots/
```

The plotting script writes:

```text
results/plots/distance_trends/
results/plots/relative_decline/
```

## Notes

- The directory name is currently `statistcs` to match the existing project code.
- SVM uses a precomputed kernel derived from the distance matrix.
- Pairwise distance computation uses `aeon.distances.pairwise_distance` when available, with a slower fallback loop for unsupported cases.
