from pathlib import Path
import time

import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_FILE = BASE_DIR / "data" / "raw" / "pima.csv"
TARGET_FILE = BASE_DIR / "data" / "raw" / "pima_target.csv"

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = RESULTS_DIR / "pima_xgboost_results.csv"


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
def load_pima():
    X = pd.read_csv(FEATURE_FILE)
    y = pd.read_csv(TARGET_FILE)

    if y.shape[1] != 1:
        raise ValueError(
            f"Expected target file to contain one column, found {y.shape[1]}"
        )

    y = y.iloc[:, 0]

    # Convert actual Pima labels into binary values
    y = y.astype(str).str.strip().str.lower()

    label_mapping = {
        "tested_negative": 0,
        "tested_positive": 1,
    }

    unexpected = set(y.unique()) - set(label_mapping.keys())

    if unexpected:
        raise ValueError(
            f"Unexpected target labels found: {sorted(unexpected)}"
        )

    y = y.map(label_mapping).astype(int)

    return X, y


# ---------------------------------------------------------
# Mathematical Macro-F1 verification
# ---------------------------------------------------------
def calculate_macro_f1_mathematically(cm):
    """
    Calculate F1 for class 0 and class 1 manually,
    then calculate Macro-F1.
    """

    if cm.shape != (2, 2):
        raise ValueError("Expected a 2x2 confusion matrix.")

    tn, fp, fn, tp = cm.ravel()

    # Class 0
    tp_0 = tn
    fp_0 = fn
    fn_0 = fp

    precision_0 = (
        tp_0 / (tp_0 + fp_0)
        if (tp_0 + fp_0) != 0
        else 0.0
    )

    recall_0 = (
        tp_0 / (tp_0 + fn_0)
        if (tp_0 + fn_0) != 0
        else 0.0
    )

    f1_0 = (
        2 * precision_0 * recall_0 / (precision_0 + recall_0)
        if (precision_0 + recall_0) != 0
        else 0.0
    )

    # Class 1
    tp_1 = tp
    fp_1 = fp
    fn_1 = fn

    precision_1 = (
        tp_1 / (tp_1 + fp_1)
        if (tp_1 + fp_1) != 0
        else 0.0
    )

    recall_1 = (
        tp_1 / (tp_1 + fn_1)
        if (tp_1 + fn_1) != 0
        else 0.0
    )

    f1_1 = (
        2 * precision_1 * recall_1 / (precision_1 + recall_1)
        if (precision_1 + recall_1) != 0
        else 0.0
    )

    macro_f1 = (f1_0 + f1_1) / 2

    return f1_0, f1_1, macro_f1


# ---------------------------------------------------------
# Build XGBoost pipeline
# ---------------------------------------------------------
def build_pipeline(resampler_name):
    classifier = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss",
    use_label_encoder=False,
)

    # No resampling
    if resampler_name == "none":
        return SklearnPipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )

    # Resampling strategies
    resampler_map = {
        "ros": RandomOverSampler(random_state=42),
        "rus": RandomUnderSampler(random_state=42),
        "smote": SMOTE(random_state=42),
        "adasyn": ADASYN(random_state=42),
        "smote_tomek": SMOTETomek(random_state=42),
        "smote_enn": SMOTEENN(random_state=42),
    }

    if resampler_name not in resampler_map:
        raise ValueError(
            f"Unknown resampling strategy: {resampler_name}"
        )

    resampler = resampler_map[resampler_name]

    # IMPORTANT:
    # Resampling happens inside the pipeline, so it is applied
    # only to the training portion of each CV fold.
    return ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("resampler", resampler),
            ("classifier", classifier),
        ]
    )


# ---------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------
def main():
    X, y = load_pima()

    strategies = [
        ("none", "No Resampling"),
        ("ros", "Random Oversampling"),
        ("rus", "Random Undersampling"),
        ("smote", "SMOTE"),
        ("adasyn", "ADASYN"),
        ("smote_tomek", "SMOTE-Tomek"),
        ("smote_enn", "SMOTE-ENN"),
    ]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    print("=" * 75)
    print("ResampleRank - Pima Resampling Benchmark")
    print("=" * 75)

    print("\nDataset:")
    print("Pima Indians Diabetes")

    print("\nClassifier:")
    print("XGBoost")

    print("\nValidation:")
    print("5-Fold Stratified Cross-Validation")

    print("\nDataset shape:")
    print(f"Samples  : {X.shape[0]}")
    print(f"Features : {X.shape[1]}")

    for strategy_code, strategy_name in strategies:

        print("\n" + "-" * 75)
        print(f"Running: {strategy_name}")
        print("-" * 75)

        model = build_pipeline(strategy_code)

        start_time = time.perf_counter()

        y_pred = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method="predict",
        )

        elapsed_time = time.perf_counter() - start_time

        # -------------------------------------------------
        # Confusion matrix
        # -------------------------------------------------
        cm = confusion_matrix(
            y,
            y_pred,
            labels=[0, 1],
        )

        tn, fp, fn, tp = cm.ravel()

        # -------------------------------------------------
        # Code metrics
        # -------------------------------------------------
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

        f1_code = f1_score(
            y,
            y_pred,
            average="macro",
            zero_division=0,
        )

        # -------------------------------------------------
        # Mathematical verification
        # -------------------------------------------------
        f1_class_0_math, f1_class_1_math, f1_math = (
            calculate_macro_f1_mathematically(cm)
        )

        metric_error = abs(f1_math - f1_code)

        status = (
            "PASS"
            if metric_error < 1e-6
            else "FAIL"
        )

        # -------------------------------------------------
        # Display
        # -------------------------------------------------
        print("Confusion Matrix:")
        print(cm)

        print(f"\nTN = {tn}")
        print(f"FP = {fp}")
        print(f"FN = {fn}")
        print(f"TP = {tp}")

        print("\nMetrics:")
        print(f"Accuracy          : {accuracy:.6f}")
        print(
            f"Balanced Accuracy : {balanced_accuracy:.6f}"
        )
        print(f"Precision         : {precision:.6f}")
        print(f"Recall            : {recall:.6f}")
        print(f"Macro-F1 (code)   : {f1_code:.6f}")

        print("\nMathematical Verification:")
        print(f"F1 Class 0        : {f1_class_0_math:.6f}")
        print(f"F1 Class 1        : {f1_class_1_math:.6f}")
        print(f"Macro-F1 (math)   : {f1_math:.6f}")
        print(f"Absolute Error    : {metric_error:.12f}")
        print(f"Verification      : {status}")
        print(f"Execution Time    : {elapsed_time:.6f} seconds")

        results.append(
            {
                "dataset": "pima",
                "resampling_code": strategy_code,
                "resampling": strategy_name,
                "classifier": "xgboost",
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
                "macro_f1_code": f1_code,
                "macro_f1_math": f1_math,
                "metric_error": metric_error,
                "verification": status,
                "execution_time_seconds": elapsed_time,
            }
        )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="macro_f1_math",
        ascending=False,
    )

    results_df.to_csv(
        RESULT_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------
    print("\n")
    print("=" * 75)
    print("FINAL RESAMPLING COMPARISON")
    print("=" * 75)

    print(
        results_df[
            [
                "resampling",
                "macro_f1_math",
                "accuracy",
                "balanced_accuracy",
                "execution_time_seconds",
                "verification",
            ]
        ].to_string(index=False)
    )

    print("\nResults saved to:")
    print(RESULT_FILE)


if __name__ == "__main__":
    main()