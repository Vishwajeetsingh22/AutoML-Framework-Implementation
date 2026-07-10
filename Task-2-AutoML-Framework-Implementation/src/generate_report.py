"""
generate_report.py
-------------------
Runs the AutoML engine on every configured dataset and writes a
plain-text summary report to outputs/automl_report.txt in the format:

    Dataset: diabetes
    Best Model = Random Forest
    Accuracy = 74.7%

A machine-readable outputs/automl_report.json is also produced with
the full per-model accuracy breakdown for every dataset, useful for
further analysis or dashboards.
"""

import json
import os

from automl import run_all_datasets

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

TXT_REPORT_PATH = os.path.join(OUTPUTS_DIR, "automl_report.txt")
JSON_REPORT_PATH = os.path.join(OUTPUTS_DIR, "automl_report.json")


def build_reports(summary: dict):
    lines = ["AutoML Framework — Summary Report", "=" * 40, ""]

    json_summary = {}
    for dataset_name, result in summary.items():
        lines.append(f"Dataset: {dataset_name}")
        lines.append(f"Best Model = {result['best_model']}")
        lines.append(f"Accuracy = {result['best_accuracy'] * 100:.1f}%")
        lines.append("")
        lines.append("All model accuracies:")
        for model_name, acc in result["all_accuracies"].items():
            lines.append(f"  - {model_name}: {acc * 100:.1f}%")
        lines.append("-" * 40)
        lines.append("")

        json_summary[dataset_name] = {
            "best_model": result["best_model"],
            "best_accuracy_pct": round(result["best_accuracy"] * 100, 2),
            "all_accuracies_pct": {
                m: round(a * 100, 2) for m, a in result["all_accuracies"].items()
            },
            "model_path": os.path.relpath(result["model_path"], BASE_DIR),
            "chart_path": os.path.relpath(result["chart_path"], BASE_DIR),
        }

    with open(TXT_REPORT_PATH, "w") as f:
        f.write("\n".join(lines))

    with open(JSON_REPORT_PATH, "w") as f:
        json.dump(json_summary, f, indent=2)

    print(f"Text report saved to:  {TXT_REPORT_PATH}")
    print(f"JSON report saved to:  {JSON_REPORT_PATH}")


if __name__ == "__main__":
    summary = run_all_datasets()
    build_reports(summary)

    print("\n" + "=" * 40)
    print("FINAL RESULTS")
    print("=" * 40)
    for dataset_name, result in summary.items():
        print(f"\n[{dataset_name}]")
        print(f"Best Model = {result['best_model']}")
        print(f"Accuracy = {result['best_accuracy'] * 100:.1f}%")
