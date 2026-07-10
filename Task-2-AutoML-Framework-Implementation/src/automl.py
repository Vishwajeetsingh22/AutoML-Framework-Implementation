"""
automl.py
---------
Core AutoML engine.

For any dataset registered in `data_preprocessing.DATASET_CONFIGS`, this
module will:

  1. Preprocess the data (via data_preprocessing.preprocess)
  2. Train 5 candidate models:
       - Logistic Regression
       - Random Forest
       - Decision Tree
       - Support Vector Machine (SVM)
       - K-Nearest Neighbors (KNN)
  3. Compare their test-set accuracy
  4. Automatically select the best model
  5. Save an accuracy-comparison chart to screenshots/
  6. Save the winning model (+ scaler + label encoder + feature names)
     to outputs/best_model_<dataset>.pkl

Run directly to execute the full AutoML workflow on every configured
dataset (Diabetes, Iris, Loan):

    python automl.py
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

from data_preprocessing import preprocess, DATASET_CONFIGS

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def get_candidate_models():
    """Return a fresh dict of the 5 candidate models (avoids state leakage across runs)."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
    }


def plot_accuracy_comparison(dataset_name: str, results: dict, save_path: str):
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] * 100 for n in names]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, accuracies, color=colors[: len(names)])
    plt.ylabel("Accuracy (%)")
    plt.title(f"AutoML Model Comparison — {dataset_name.capitalize()} Dataset")
    plt.ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                  f"{acc:.1f}%", ha="center", fontweight="bold")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def run_automl(dataset_name: str, verbose: bool = True) -> dict:
    """
    Run the full AutoML workflow for a single dataset.

    Returns a summary dict with per-model accuracy, the best model's
    name/accuracy, and the path the winning model was saved to.
    """
    X_train, X_test, y_train, y_test, scaler, feature_names, label_encoder = preprocess(
        dataset_name
    )

    models = get_candidate_models()
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = {"model": model, "accuracy": acc}
        if verbose:
            print(f"  {name:22s} accuracy = {acc:.4f}")

    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_model = results[best_name]["model"]
    best_acc = results[best_name]["accuracy"]

    if verbose:
        print(f"  -> Best Model = {best_name} | Accuracy = {best_acc * 100:.1f}%")
        y_pred_best = best_model.predict(X_test)
        print(classification_report(y_test, y_pred_best, zero_division=0))

    # Save comparison chart
    chart_path = os.path.join(SCREENSHOTS_DIR, f"accuracy_{dataset_name}.png")
    plot_accuracy_comparison(dataset_name, results, chart_path)

    # Save the winning model artifact
    artifact = {
        "dataset": dataset_name,
        "model": best_model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "model_name": best_name,
        "test_accuracy": best_acc,
        "all_results": {n: r["accuracy"] for n, r in results.items()},
    }
    model_path = os.path.join(OUTPUTS_DIR, f"best_model_{dataset_name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    return {
        "dataset": dataset_name,
        "all_accuracies": {n: r["accuracy"] for n, r in results.items()},
        "best_model": best_name,
        "best_accuracy": best_acc,
        "model_path": model_path,
        "chart_path": chart_path,
    }


def run_all_datasets():
    summary = {}
    for dataset_name in DATASET_CONFIGS:
        print(f"\n=== Running AutoML on '{dataset_name}' dataset ===")
        summary[dataset_name] = run_automl(dataset_name)
    return summary


if __name__ == "__main__":
    summary = run_all_datasets()

    print("\n" + "=" * 50)
    print("AUTOML SUMMARY")
    print("=" * 50)
    for dataset_name, result in summary.items():
        print(f"\nDataset: {dataset_name}")
        print(f"Best Model = {result['best_model']}")
        print(f"Accuracy = {result['best_accuracy'] * 100:.1f}%")
