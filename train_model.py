
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

for folder in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "ai4i_dataset.csv"
MODEL_PATH = MODELS_DIR / "machinasense.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset missing. Run the dataset download step first."
        )

    df = pd.read_csv(DATA_PATH)

    target = "Machine_failure"
    if target not in df.columns:
        raise ValueError(f"Missing target column: {target}")

    # Exclude identifiers and failure-mode labels from the features.
    excluded = {
        "UDI", "Product_ID", target,
        "TWF", "HDF", "PWF", "OSF", "RNF"
    }

    feature_cols = [
        c for c in df.columns
        if c not in excluded
    ]

    X = df[feature_cols].copy()
    y = pd.to_numeric(df[target], errors="raise").astype(int)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Machine_failure must contain binary labels.")

    # Chronological split: 70% train, 15% validation, 15% test.
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [
        c for c in feature_cols if c not in numeric_cols
    ]

    preprocessing = ColumnTransformer([
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_cols,
        ),
    ])

    classifier = Pipeline([
        ("preprocessing", preprocessing),
        ("model", RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("Training failure classifier...")
    classifier.fit(X_train, y_train)

    # Select the classification threshold using validation data only.
    val_probabilities = classifier.predict_proba(X_val)[:, 1]
    thresholds = [i / 100 for i in range(10, 91, 5)]

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        predictions = (val_probabilities >= threshold).astype(int)
        score = f1_score(y_val, predictions, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    # Fit anomaly detector on normal training observations only.
    normal_train = X_train.loc[y_train == 0, numeric_cols]

    anomaly_scaler = StandardScaler()
    normal_scaled = anomaly_scaler.fit_transform(normal_train)

    anomaly_detector = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    anomaly_detector.fit(normal_scaled)

    # Evaluate once on the untouched test partition.
    test_probabilities = classifier.predict_proba(X_test)[:, 1]
    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    cm = confusion_matrix(
        y_test, test_predictions, labels=[0, 1]
    )

    metrics = {
        "dataset": "UCI AI4I 2020",
        "dataset_is_synthetic": False,
        "split": "chronological 70/15/15",
        "train_rows": len(X_train),
        "validation_rows": len(X_val),
        "test_rows": len(X_test),
        "threshold": float(best_threshold),
        "accuracy": float(
            accuracy_score(y_test, test_predictions)
        ),
        "precision": float(
            precision_score(y_test, test_predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, test_predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_test, test_predictions, zero_division=0)
        ),
        "average_precision": float(
            average_precision_score(y_test, test_probabilities)
        ),
        "false_positives": int(cm[0, 1]),
        "confusion_matrix": cm.tolist(),
    }

    bundle = {
        "classifier": classifier,
        "anomaly_detector": anomaly_detector,
        "anomaly_scaler": anomaly_scaler,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "threshold": float(best_threshold),
    }

    joblib.dump(bundle, MODEL_PATH)

    test_results = X_test.copy()
    test_results["actual_failure"] = y_test
    test_results["failure_probability"] = test_probabilities
    test_results["predicted_failure"] = test_predictions
    test_results["anomaly_score"] = -anomaly_detector.decision_function(
        anomaly_scaler.transform(X_test[numeric_cols])
    )
    test_results.to_csv(
        REPORTS_DIR / "test_predictions.csv", index=False
    )

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nMODEL EVALUATION")
    print(json.dumps(metrics, indent=2))
    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")


if __name__ == "__main__":
    main()
