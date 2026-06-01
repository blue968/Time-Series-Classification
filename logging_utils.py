import csv
from pathlib import Path

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

    def log_result(self, dataset, noise_type, intensity, accuracy, classifier="1-NN", metric=None):
        classifier_col = self._classifier_column(classifier)
        entry = {
            "Dataset": dataset,
            "Noise_Type": noise_type,
            "Intensity": intensity,
            "Avg_Accuracy": accuracy,
            "Classifier": classifier_col,
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
        import pandas as pd

        return pd.DataFrame(self.raw_data)
