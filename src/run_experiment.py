from pathlib import Path
import argparse
import time

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


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(dataset_name):
    feature_file = RAW_DIR / f"{dataset_name}.csv"
    target_file = RAW_DIR / f"{dataset_name}_target.csv"

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feature_file}"
        )

    if not target_file.exists():
        raise FileNotFoundError(
            f"Target file not found: {target_file}"
        )

    X = pd.read_csv(feature_file)
    y = pd.read_csv(target_file)

    if y.shape[1] != 1:
        raise ValueError(
            f"Expected target file to contain one column, "
            f"found {y.shape[1]}"
        )

    y = y.iloc[:, 0].astype(str).str.strip()

    unique_classes = y.unique()

    if len(unique_classes) != 2:
        raise ValueError(
            f"Expected exactly 2 classes, found {len(unique_classes)}: "
            f"{sorted(unique_classes)}"
        )

    class_counts = y.value_counts()

    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    label_mapping = {
        majority_class: 0,
        minority_class: 1,
    }

    y = y.map(label_mapping).astype(int)

    return X, y, majority_class, minority_class


def manual_f1_from_confusion_matrix(cm):
    if cm.shape != (2, 2):
        raise ValueError(
            "Expected a 2x2 confusion matrix."
        )

    tn, fp, fn, tp = cm.ravel()

    def f1_from_values(tp_value, fp_value, fn_value):
        precision_den = tp_value + fp_value
        recall_den = tp_value + fn_value

        precision = (
            tp_value / precision_den
            if precision_den != 0
            else 0.0
        )

        recall = (
            tp_value / recall_den
            if recall_den != 0
            else 0.0
        )

        if precision + recall == 0:
            return 0.0

        return (
            2 * precision * recall
            / (precision + recall)
        )

    f1_class_0 = f1_from_values(
        tn,
        fn,
        fp,
    )

    f1_class_1 = f1_from_values(
        tp,
        fp,
        fn,
    )

    macro_f1 = (
        f1_class_0 + f1_class_1
    ) / 2

    return (
        f1_class_0,
        f1_class_1,
        macro_f1,
    )


def run_experiment(dataset_name):
    X, y, majority_class, minority_class = load_dataset(
        dataset_name
    )

    print("=" * 70)
    print("ResampleRank - Dataset-Aware Baseline")
    print("=" * 70)

    print(f"\nDataset           : {dataset_name}")
    print("Resampling        : None")
    print("Classifier        : Logistic Regression")
    print("Validation        : 5-Fold Stratified Cross-Validation")

    print("\nClass mapping:")
    print(f"Majority class    : {majority_class} -> 0")
    print(f"Minority class    : {minority_class} -> 1")

    print("\nDataset shape:")
    print(f"Samples           : {X.shape[0]}")
    print(f"Features          : {X.shape[1]}")

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    start_time = time.perf_counter()

    y_pred = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict",
    )

    elapsed_time = (
        time.perf_counter() - start_time
    )

    cm = confusion_matrix(
        y,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    macro_f1_code = f1_score(
        y,
        y_pred,
        average="macro",
        zero_division=0,
    )

    accuracy = accuracy_score(
        y,
        y_pred,
    )

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

    (
        f1_class_0_math,
        f1_class_1_math,
        macro_f1_math,
    ) = manual_f1_from_confusion_matrix(cm)

    metric_error = abs(
        macro_f1_math - macro_f1_code
    )

    print("\n" + "-" * 70)
    print("RESULT")
    print("-" * 70)

    print("\nConfusion Matrix:")
    print(cm)

    print(f"\nTN                : {tn}")
    print(f"FP                : {fp}")
    print(f"FN                : {fn}")
    print(f"TP                : {tp}")

    print(f"\nAccuracy          : {accuracy:.6f}")
    print(
        f"Balanced Accuracy : "
        f"{balanced_accuracy:.6f}"
    )
    print(f"Precision         : {precision:.6f}")
    print(f"Recall            : {recall:.6f}")
    print(
        f"Macro-F1 (code)   : "
        f"{macro_f1_code:.6f}"
    )

    print("\nMathematical Verification:")
    print(
        f"F1 Class 0        : "
        f"{f1_class_0_math:.6f}"
    )
    print(
        f"F1 Class 1        : "
        f"{f1_class_1_math:.6f}"
    )
    print(
        f"Macro-F1 (math)   : "
        f"{macro_f1_math:.6f}"
    )
    print(
        f"Absolute Error    : "
        f"{metric_error:.12f}"
    )

    if metric_error < 1e-6:
        verification = "PASS"
        print("\nSTATUS: PASS")
    else:
        verification = "FAIL"
        print("\nSTATUS: FAIL")

    print(
        f"\nExecution Time    : "
        f"{elapsed_time:.6f} seconds"
    )

    result_file = (
        RESULTS_DIR
        / f"{dataset_name}_baseline_results.csv"
    )

    result = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
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
                "verification": verification,
                "execution_time_seconds": elapsed_time,
            }
        ]
    )

    result.to_csv(
        result_file,
        index=False,
    )

    print("\nResult saved to:")
    print(result_file)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the ResampleRank "
            "dataset-aware baseline experiment."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, for example pima or glass1.",
    )

    args = parser.parse_args()

    run_experiment(args.dataset)


if __name__ == "__main__":
    main()