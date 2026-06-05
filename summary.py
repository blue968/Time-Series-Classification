"""
Aggregate all accuracy.csv files under results/statistcs/ into a single summary table.

Output columns: dataset, metric, classifier, noise_type, intensity, accuracy
"""

import csv
from pathlib import Path


def aggregate_results(stat_dir: str = "results/statistcs", output_path: str = "results/summary.csv"):
    """Walk stat_dir, read every accuracy.csv, and merge into one summary CSV."""
    rows = []
    stat_root = Path(stat_dir)

    if not stat_root.exists():
        print(f"Directory not found: {stat_root.resolve()}")
        return

    csv_files = sorted(stat_root.rglob("accuracy.csv"))
    print(f"Found {len(csv_files)} accuracy.csv files")

    for csv_path in csv_files:
        # Path structure: results/statistcs/{category}/{dataset}/{distance_family}/{metric}/accuracy.csv
        # Parse from the right: metric=parts[-2], family=parts[-3], dataset=parts[-4], category=parts[-5]
        parts = csv_path.parts
        metric = parts[-2]
        dataset = parts[-4]
        # category = parts[-5]  # not used in output per user spec

        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                noise_type = row.get("Noise_Type", "").strip()
                intensity = row.get("Intensity", "").strip()
                # Melt classifier columns (1nn, svm) into rows
                for classifier in ("1nn", "svm"):
                    accuracy = row.get(classifier, "").strip()
                    if accuracy:
                        rows.append({
                            "dataset": dataset,
                            "metric": metric,
                            "classifier": classifier,
                            "noise_type": noise_type,
                            "intensity": intensity,
                            "accuracy": accuracy,
                        })

    # Write summary
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["dataset", "metric", "classifier", "noise_type", "intensity", "accuracy"]
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output.resolve()}")
    # Print a quick preview of unique values per column
    datasets = sorted(set(r["dataset"] for r in rows))
    metrics = sorted(set(r["metric"] for r in rows))
    classifiers = sorted(set(r["classifier"] for r in rows))
    noise_types = sorted(set(r["noise_type"] for r in rows))
    print(f"  datasets: {len(datasets)}")
    print(f"  metrics: {len(metrics)}")
    print(f"  classifiers: {classifiers}")
    print(f"  noise_types: {noise_types}")


if __name__ == "__main__":
    aggregate_results()
