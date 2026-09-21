from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_FILE = BASE_DIR / "data" / "raw" / "pima.csv"
TARGET_FILE = BASE_DIR / "data" / "raw" / "pima_target.csv"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = RESULTS_DIR / "benchmark_results.csv"


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
def load_pima():
    X = pd.read_csv(FEATURE_FILE)
    y = pd.read_csv(TARGET_FILE)

    # Target file should contain one column
    if y.shape[1] != 1:
        raise ValueError(
            f"Expected target file to contain one column, found {y.shape[1]}"
        )

    y = y.iloc[:, 0]

    # Convert the actual Pima labels to binary values
    y = y.astype(str).str.strip().str.lower()

    label_mapping = {
        "tested_negative": 0,
        "tested_positive": 1,
    }

    if not set(y.unique()).issubset(label_mapping.keys()):
        raise ValueError(
            f"Unexpected target labels found: {sorted(y.unique())}"
        )

    y = y.map(label_mapping).astype(int)

    return X, y


# ---------------------------------------------------------
# Manual binary F1 calculation
# ---------------------------------------------------------
def manual_f1_from_confusion_matrix(cm):
    """
    Calculate F1 for each class manually from TP, FP, FN,
    then calculate Macro-F1.
    """

    if cm.shape != (2, 2):
        raise ValueError("Expected a 2x2 confusion matrix.")

    tn, fp, fn, tp = cm.ravel()

    def f1_from_values(tp_value, fp_value, fn_value):
        precision_den = tp_value + fp_value
        recall_den = tp_value + fn_value

        precision = (
            tp_value / precision_den if precision_den != 0 else 0.0
        )

        recall = (
            tp_value / recall_den if recall_den != 0 else 0.0
        )

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    # Class 0:
    # TP_0 = TN
    # FP_0 = FN
    # FN_0 = FP
    f1_class_0 = f1_from_values(
        tn,
        fn,
        fp,
    )

    # Class 1:
    f1_class_1 = f1_from_values(
        tp,
        fp,
        fn,
    )

    macro_f1 = (f1_class_0 + f1_class_1) / 2

    return f1_class_0, f1_class_1, macro_f1


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------
def main():
    X, y = load_pima()

    print("=" * 60)
    print("ResampleRank - First Benchmark Experiment")
    print("=" * 60)

    print("\nDataset:")
    print("Pima Indians Diabetes")

    print("\nExperiment:")
    print("Resampling : None")
    print("Classifier : Logistic Regression")
    print("Validation : 5-Fold Stratified Cross-Validation")

    print("\nDataset shape:")
    print(f"Samples  : {X.shape[0]}")
    print(f"Features : {X.shape[1]}")

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    # -----------------------------------------------------
    # Cross-validation
    # -----------------------------------------------------
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    start_time = time.perf_counter()

    # Out-of-fold predictions
    y_pred = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict",
    )

    elapsed_time = time.perf_counter() - start_time

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------
    cm = confusion_matrix(
        y,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    # -----------------------------------------------------
    # Sklearn metrics
    # -----------------------------------------------------
    macro_f1_code = f1_score(
        y,
        y_pred,
        average="macro",
        zero_division=0,
    )

    accuracy = accuracy_score(y, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y,
        y_pred,
    )

    precision = precision_score(
        y,
        y_pred,
        average="binary",
        zero_division=0,
    )

    recall = recall_score(
        y,
        y_pred,
        average="binary",
        zero_division=0,
    )

    # -----------------------------------------------------
    # Independent mathematical calculation
    # -----------------------------------------------------
    f1_class_0_math, f1_class_1_math, macro_f1_math = (
        manual_f1_from_confusion_matrix(cm)
    )

    metric_error = abs(
        macro_f1_math - macro_f1_code
    )

    # -----------------------------------------------------
    # Display results
    # -----------------------------------------------------
    print("\n" + "-" * 60)
    print("CONFUSION MATRIX")
    print("-" * 60)

    print(cm)

    print("\nTN =", tn)
    print("FP =", fp)
    print("FN =", fn)
    print("TP =", tp)

    print("\n" + "-" * 60)
    print("SKLEARN RESULTS")
    print("-" * 60)

    print(f"Accuracy          : {accuracy:.6f}")
    print(f"Balanced Accuracy : {balanced_accuracy:.6f}")
    print(f"Precision         : {precision:.6f}")
    print(f"Recall            : {recall:.6f}")
    print(f"Macro-F1 (code)   : {macro_f1_code:.6f}")

    print("\n" + "-" * 60)
    print("MATHEMATICAL VERIFICATION")
    print("-" * 60)

    print(f"F1 Class 0        : {f1_class_0_math:.6f}")
    print(f"F1 Class 1        : {f1_class_1_math:.6f}")
    print(f"Macro-F1 (math)   : {macro_f1_math:.6f}")
    print(f"Absolute Error    : {metric_error:.12f}")

    if metric_error < 1e-6:
        print("\nSTATUS: PASS")
        print("Mathematical result matches the Python result.")
    else:
        print("\nSTATUS: FAIL")
        print("Mathematical result does not match the Python result.")

    print(f"\nExecution Time    : {elapsed_time:.6f} seconds")

    # -----------------------------------------------------
    # Save result
    # -----------------------------------------------------
    result = pd.DataFrame(
        [
            {
                "dataset": "pima",
                "resampling": "none",
                "classifier": "logistic_regression",
                "samples": X.shape[0],
                "features": X.shape[1],
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "precision": precision,
                "recall": recall,
                "f1_class_0_math": f1_class_0_math,
                "f1_class_1_math": f1_class_1_math,
                "macro_f1_code": macro_f1_code,
                "macro_f1_math": macro_f1_math,
                "metric_error": metric_error,
                "execution_time_seconds": elapsed_time,
            }
        ]
    )

    result.to_csv(
        RESULT_FILE,
        index=False,
    )

    print("\nResult saved to:")
    print(RESULT_FILE)


if __name__ == "__main__":
    main()