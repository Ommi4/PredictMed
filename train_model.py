from pathlib import Path
import json

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

DATA_FILE = Path("data/processed_patients.csv")
MODEL_DIR = Path("models")

TARGET = "future_egfr_below_30"

NUMERIC_FEATURES = [
    "early_egfr_first",
    "early_egfr_last",
    "early_egfr_mean",
    "early_egfr_min",
    "early_egfr_slope_year",
    "early_visit_count",
    "days_observed",
    "blgfr",
]

CATEGORICAL_FEATURES = [
    "trt01pn",
    "blglp1",
    "strata",
]


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Run python data_pipeline.py first."
        )

    data = pd.read_csv(DATA_FILE)

    required = set(
        NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    )

    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = data.dropna(subset=[TARGET]).copy()

    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET].astype(int)

    print("Total patients:", len(data))
    print("Outcome counts:")
    print(y.value_counts())

    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError(
            "The target does not contain enough examples of both "
            "classes. Check the data and outcome definition before "
            "training a classifier."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "test_patients": int(len(y_test)),
        "positive_test_cases": int(y_test.sum()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "roc_auc": (
            float(roc_auc_score(y_test, probabilities))
            if y_test.nunique() == 2
            else None
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "target": "Any observed follow-up eGFR below 30 after day 365",
    }

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(model, MODEL_DIR / "predictmed_model.joblib")

    with open(MODEL_DIR / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    test_results = data.loc[X_test.index, ["usubjid", TARGET]].copy()
    test_results["predicted_class"] = predictions
    test_results["estimated_probability"] = probabilities
    test_results.to_csv(
        MODEL_DIR / "test_predictions.csv", index=False
    )

    print("\nEvaluation on held-out patients:")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    print("\nSaved model to models/predictmed_model.joblib")


if __name__ == "__main__":
    main()