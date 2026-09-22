from pathlib import Path
import argparse
import time

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


RESAMPLING_STRATEGIES = [
    ("none", "No Resampling"),
    ("ros", "Random Oversampling"),
    ("rus", "Random Undersampling"),
    ("smote", "SMOTE"),
    ("adasyn", "ADASYN"),
    ("smote_tomek", "SMOTE-Tomek"),
    ("smote_enn", "SMOTE-ENN"),
]


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
    y_df = pd.read_csv(target_file)

    if y_df.shape[1] != 1:
        raise ValueError(
            f"Expected one target column, found {y_df.shape[1]}"
        )

    y = y_df.iloc[:, 0].astype(str).str.strip()

    class_counts = y.value_counts()

    if len(class_counts) != 2:
        raise ValueError(
            f"Expected binary classification, found "
            f"{len(class_counts)} classes."
        )

    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    label_mapping = {
        majority_class: 0,
        minority_class: 1,
    }

    y = y.map(label_mapping).astype(int)

    return X, y, majority_class, minority_class


def calculate_macro_f1_mathematically(cm):
    if cm.shape != (2, 2):
        raise ValueError(
            "Expected a 2x2 confusion matrix."
        )

    tn, fp, fn, tp = cm.ravel()

    def calculate_f1(tp_value, fp_value, fn_value):
        precision_denominator = tp_value + fp_value
        recall_denominator = tp_value + fn_value

        precision = (
            tp_value / precision_denominator
            if precision_denominator != 0
            else 0.0
        )

        recall = (
            tp_value / recall_denominator
            if recall_denominator != 0
            else 0.0
        )

        if precision + recall == 0:
            return 0.0

        return (
            2 * precision * recall
            / (precision + recall)
        )

    f1_class_0 = calculate_f1(
        tn,
        fn,
        fp,
    )

    f1_class_1 = calculate_f1(
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


def build_classifier(classifier_name):
    if classifier_name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=42,
        )

    if classifier_name == "decision_tree":
        return DecisionTreeClassifier(
            random_state=42,
        )

    if classifier_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )

    if classifier_name == "xgboost":
        return XGBClassifier(
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

    raise ValueError(
        f"Unknown classifier: {classifier_name}"
    )


def build_pipeline(
    resampling_code,
    classifier_name,
):
    classifier = build_classifier(
        classifier_name
    )

    if resampling_code == "none":
        return SklearnPipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )

    resampler_map = {
        "ros": RandomOverSampler(
            random_state=42
        ),
        "rus": RandomUnderSampler(
            random_state=42
        ),
        "smote": SMOTE(
            random_state=42
        ),
        "adasyn": ADASYN(
            random_state=42
        ),
        "smote_tomek": SMOTETomek(
            random_state=42
        ),
        "smote_enn": SMOTEENN(
            random_state=42
        ),
    }

    if resampling_code not in resampler_map:
        raise ValueError(
            f"Unknown resampling strategy: "
            f"{resampling_code}"
        )

    return ImbPipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "resampler",
                resampler_map[resampling_code],
            ),
            ("classifier", classifier),
        ]
    )


def run_classifier_benchmark(
    dataset_name,
    X,
    y,
    classifier_name,
):
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    print("\n")
    print("=" * 75)
    print(
        f"{dataset_name} - "
        f"{classifier_name}"
    )
    print("=" * 75)

    for (
        strategy_code,
        strategy_name,
    ) in RESAMPLING_STRATEGIES:

        print(
            f"\nRunning: {strategy_name}"
        )

        model = build_pipeline(
            strategy_code,
            classifier_name,
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
            time.perf_counter()
            - start_time
        )

        cm = confusion_matrix(
            y,
            y_pred,
            labels=[0, 1],
        )

        tn, fp, fn, tp = cm.ravel()

        accuracy = accuracy_score(
            y,
            y_pred,
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y,
                y_pred,
            )
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

        macro_f1_code = f1_score(
            y,
            y_pred,
            average="macro",
            zero_division=0,
        )

        (
            f1_class_0_math,
            f1_class_1_math,
            macro_f1_math,
        ) = calculate_macro_f1_mathematically(
            cm
        )

        metric_error = abs(
            macro_f1_math
            - macro_f1_code
        )

        verification = (
            "PASS"
            if metric_error < 1e-6
            else "FAIL"
        )

        print(
            f"Macro-F1          : "
            f"{macro_f1_math:.6f}"
        )

        print(
            f"Verification      : "
            f"{verification}"
        )

        print(
            f"Execution Time    : "
            f"{elapsed_time:.6f} seconds"
        )

        results.append(
            {
                "dataset": dataset_name,
                "resampling_code": strategy_code,
                "resampling": strategy_name,
                "classifier": classifier_name,
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
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="macro_f1_math",
        ascending=False,
    ).reset_index(drop=True)

    result_file = (
        RESULTS_DIR
        / f"{dataset_name}_{classifier_name}_results.csv"
    )

    results_df.to_csv(
        result_file,
        index=False,
    )

    print(
        f"\nSaved: {result_file}"
    )

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete ResampleRank "
            "benchmark for one dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Dataset name, such as "
            "pima or glass1."
        ),
    )

    args = parser.parse_args()

    dataset_name = args.dataset

    X, y, majority_class, minority_class = (
        load_dataset(dataset_name)
    )

    print("=" * 75)
    print("ResampleRank - Full Dataset Benchmark")
    print("=" * 75)

    print(
        f"\nDataset           : "
        f"{dataset_name}"
    )

    print(
        f"Samples           : "
        f"{X.shape[0]}"
    )

    print(
        f"Features          : "
        f"{X.shape[1]}"
    )

    print(
        f"Majority class    : "
        f"{majority_class} -> 0"
    )

    print(
        f"Minority class    : "
        f"{minority_class} -> 1"
    )

    classifiers = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
    ]

    all_results = []

    for classifier_name in classifiers:
        result_df = run_classifier_benchmark(
            dataset_name,
            X,
            y,
            classifier_name,
        )

        all_results.append(result_df)

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    combined_file = (
        RESULTS_DIR
        / f"{dataset_name}_all_results.csv"
    )

    combined.to_csv(
        combined_file,
        index=False,
    )

    verification_failures = (
        combined["verification"] != "PASS"
    ).sum()

    print("\n")
    print("=" * 75)
    print("BENCHMARK SUMMARY")
    print("=" * 75)

    print(
        f"Total experiments : "
        f"{len(combined)}"
    )

    print(
        f"Verification failures : "
        f"{verification_failures}"
    )

    print(
        f"Macro-F1 minimum : "
        f"{combined['macro_f1_math'].min():.6f}"
    )

    print(
        f"Macro-F1 maximum : "
        f"{combined['macro_f1_math'].max():.6f}"
    )

    print(
        f"\nCombined results saved to:"
    )

    print(combined_file)


if __name__ == "__main__":
    main()