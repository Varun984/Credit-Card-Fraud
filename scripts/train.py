"""
Training script for Credit Card Fraud Detection model.
Run from project root: python scripts/train.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

# Project root (creditcard-fraud-detection)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_FILE = "PS_20174392719_1491204439457_log.csv"


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    data_path = DATA_DIR / DATA_FILE
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        sys.exit(1)

    df = pd.read_csv(data_path)
    X = df.drop(["isFraud"], axis=1)
    Y = df["isFraud"]

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    encoder.fit(X_train[categorical_cols])

    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    X_train_encoded[categorical_cols] = encoder.transform(X_train[categorical_cols])
    X_test_encoded[categorical_cols] = encoder.transform(X_test[categorical_cols])

    joblib.dump(encoder, MODELS_DIR / "ordinal_encoder.pkl")

    scale_pos_weight = len(Y_train[Y_train == 0]) / len(Y_train[Y_train == 1])
    xgb = XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    xgb.fit(X_train_encoded, Y_train)

    y_probs = xgb.predict_proba(X_test_encoded)[:, 1]
    y_pred = (y_probs >= 0.9924).astype(int)

    print(classification_report(Y_test, y_pred))
    print(confusion_matrix(Y_test, y_pred))
    print("Accuracy:", accuracy_score(Y_test, y_pred))
    print("AUC Score:", roc_auc_score(Y_test, y_probs))

    joblib.dump(xgb, MODELS_DIR / "xgb_model.pkl")
    print(f"\nModel and encoder saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
