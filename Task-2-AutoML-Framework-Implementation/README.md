# AutoML Framework Implementation

An end-to-end **AutoML (Automated Machine Learning) framework** that
automatically preprocesses any tabular classification dataset, trains
multiple candidate models, compares their accuracy, and selects the
best-performing one — demonstrated on three different datasets:
**Diabetes**, **Iris**, and **Loan**.

## 🎯 Objective

Automatically train multiple ML models on a given dataset and select
the best one, without writing dataset-specific model-selection code
each time.

## 📁 Project Structure

```
Task-2-AutoML-Framework-Implementation/
│
├── data/
│   ├── diabetes.csv                # Pima Indians Diabetes dataset (768 records)
│   ├── iris.csv                    # Iris flower dataset (150 records)
│   └── loan.csv                    # Loan approval dataset (491 records)
│
├── notebooks/
│   └── automl_analysis.ipynb       # Runs the AutoML engine on all 3 datasets (pre-run)
│
├── src/
│   ├── data_preprocessing.py       # Generic, dataset-agnostic preprocessing pipeline
│   ├── automl.py                   # Core AutoML engine: trains, compares, selects best model
│   └── generate_report.py          # Runs AutoML on all datasets + writes summary reports
│
├── screenshots/
│   ├── accuracy_diabetes.png       # Model accuracy comparison — Diabetes
│   ├── accuracy_iris.png           # Model accuracy comparison — Iris
│   └── accuracy_loan.png           # Model accuracy comparison — Loan
│
├── outputs/
│   ├── best_model_diabetes.pkl     # Winning model + scaler + feature names — Diabetes
│   ├── best_model_iris.pkl         # Winning model + scaler + feature names — Iris
│   ├── best_model_loan.pkl         # Winning model + scaler + feature names — Loan
│   ├── automl_report.txt           # Human-readable summary report
│   └── automl_report.json          # Machine-readable summary report
│
├── documentation/
│   └── project_report.pdf          # Written project report
│
├── requirements.txt
└── README.md
```

## 🔄 Workflow

```
Input Dataset
      ↓
Data Preprocessing
      ↓
Train Multiple Models
      ↓
Compare Accuracy
      ↓
Select Best Model
      ↓
Generate Report
```

## 🤖 Models Trained

- Logistic Regression
- Random Forest
- Decision Tree
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

## ⚙️ How Preprocessing Works

`src/data_preprocessing.py` is fully generic — a new dataset can be
plugged in just by adding an entry to `DATASET_CONFIGS` (path, target
column, columns to drop, and any columns where `0` really means
"missing"). For every dataset it will:

1. Drop irrelevant identifier columns (e.g. `Loan_ID`)
2. Replace implausible zero values with the column median (Diabetes)
3. Impute missing values — median for numeric columns, mode for
   categorical columns (Loan has real missing values in `Gender`,
   `Self_Employed`, `Credit_History`, etc.)
4. One-hot encode categorical predictors
5. Label-encode non-numeric targets
6. Split into train/test sets (80/20, stratified)
7. Scale numeric features with `StandardScaler`

## 📊 Datasets Used

| Dataset | Records | Features | Target | Notes |
|---|---|---|---|---|
| **Diabetes** | 768 | 8 (numeric) | Outcome (binary) | Pima Indians Diabetes dataset |
| **Iris** | 150 | 4 (numeric) | species (3 classes) | Classic multi-class benchmark |
| **Loan** | 491 | 12 (mixed) | Loan_Status (binary) | Categorical features + missing values |

## ⚙️ Setup & Installation

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ How to Run

### Option A — Run the AutoML engine on everything at once
```bash
cd src
python generate_report.py
```
This preprocesses all three datasets, trains all 5 models on each,
saves accuracy-comparison charts to `screenshots/`, saves the winning
model for each dataset to `outputs/`, and writes
`outputs/automl_report.txt` + `outputs/automl_report.json`.

### Option B — Run AutoML on a single dataset
```python
from automl import run_automl

result = run_automl("diabetes")   # or "iris" / "loan"
print(result["best_model"], result["best_accuracy"])
```

### Option C — Explore the notebook
```bash
jupyter notebook notebooks/automl_analysis.ipynb
```
The notebook is already pre-executed with saved outputs.

## 📈 Results

```
Dataset: diabetes
Best Model = Random Forest
Accuracy = 74.7%

Dataset: iris
Best Model = SVM
Accuracy = 96.7%

Dataset: loan
Best Model = Random Forest
Accuracy = 83.8%
```

Full per-model accuracy breakdowns are in `outputs/automl_report.txt`
and `outputs/automl_report.json`. See `documentation/project_report.pdf`
for the complete write-up.

## 🛠️ Tech Stack

- Python 3
- pandas, NumPy
- scikit-learn (Logistic Regression, Random Forest, Decision Tree, SVM, KNN)
- matplotlib (visualization)
- Jupyter Notebook

## 📌 Notes for Submission

- This folder is self-contained and can be zipped/pushed to GitHub as-is.
- All model files in `outputs/` are already generated — re-running
  `generate_report.py` will regenerate them deterministically
  (fixed `random_state=42` on every model and split).
- To plug in a new dataset, just add a config entry to
  `DATASET_CONFIGS` in `src/data_preprocessing.py`.
