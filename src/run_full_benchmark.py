from pathlib import Path
import argparse
import math
import time

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

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


N_SPLITS = 5
RANDOM_STATE = 42


def read_keel_feature_types(dataset_name, feature_columns):
    """
    Read the original KEEL attribute definitions and determine
    whether each feature is numeric or categorical.

    If a KEEL .dat file is unavailable, all CSV features are
    treated as numeric.
    """

    input_file = RAW_DIR / f"{dataset_name}.dat"

    if not input_file.exists():
        return {
            column: "numeric"
            for column in feature_columns
        }

    feature_types = {}

    with open(input_file, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            if not line.lower().startswith("@attribute"):
                continue

            parts = line.split(maxsplit=2)

            if len(parts) < 3:
                continue

            name = parts[1]
            definition = parts[2].strip()

            if name.lower() == "class":
                continue

            if (
                definition.startswith("{")
                and definition.endswith("}")
            ):
                feature_types[name] = "categorical"

            elif definition.lower().startswith(
                ("real", "integer")
            ):
                feature_types[name] = "numeric"

            else:
                raise ValueError(
                    f"Unsupported attribute definition "
                    f"for '{name}': {definition}"
                )

    return feature_types


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
            f"Expected one target column, found "
            f"{y_df.shape[1]}"
        )

    y = (
        y_df.iloc[:, 0]
        .astype(str)
        .str.strip()
    )

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

    feature_types = read_keel_feature_types(
        dataset_name,
        X.columns,
    )

    missing_definitions = [
        column
        for column in X.columns
        if column not in feature_types
    ]

    if missing_definitions:
        raise ValueError(
            "Missing KEEL feature definitions for columns: "
            f"{missing_definitions}"
        )

    numeric_features = [
        column
        for column in X.columns
        if feature_types[column] == "numeric"
    ]

    categorical_features = [
        column
        for column in X.columns
        if feature_types[column] == "categorical"
    ]

    for column in categorical_features:
        X[column] = (
            X[column]
            .astype(str)
            .str.strip()
        )

    for column in numeric_features:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    if X.isnull().any().any():
        missing_columns = X.columns[
            X.isnull().any()
        ].tolist()

        raise ValueError(
            f"Missing/invalid feature values in: "
            f"{missing_columns}"
        )

    return (
        X,
        y,
        majority_class,
        minority_class,
        numeric_features,
        categorical_features,
    )


def build_preprocessor(
    numeric_features,
    categorical_features,
):
    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse=False,
                ),
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError(
            "No numeric or categorical features found."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def get_safe_k_neighbors(y):
    """
    Select a k-nearest-neighbor value that is safe for
    all training folds.
    """

    class_counts = y.value_counts()

    minority_count = int(
        class_counts.min()
    )

    minimum_training_minority = (
        minority_count
        - math.ceil(
            minority_count / N_SPLITS
        )
    )

    safe_k = min(
        5,
        minimum_training_minority - 1,
    )

    safe_k = max(1, safe_k)

    return safe_k


def build_classifier(classifier_name):
    if classifier_name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        )

    if classifier_name == "decision_tree":
        return DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        )

    if classifier_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if classifier_name == "xgboost":
        return XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
            use_label_encoder=False,
        )

    raise ValueError(
        f"Unknown classifier: {classifier_name}"
    )


def build_resampler(
    resampling_code,
    y,
):
    """
    Build a fresh resampler for each experiment.

    SMOTE/ADASYN-family methods use a safe neighbor count
    so very small minority classes do not fail because of
    an invalid neighbor count.
    """

    k_neighbors = get_safe_k_neighbors(y)

    if resampling_code == "ros":
        return RandomOverSampler(
            random_state=RANDOM_STATE
        )

    if resampling_code == "rus":
        return RandomUnderSampler(
            random_state=RANDOM_STATE
        )

    if resampling_code == "smote":
        return SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=k_neighbors,
        )

    if resampling_code == "adasyn":
        return ADASYN(
            random_state=RANDOM_STATE,
            n_neighbors=k_neighbors,
        )

    if resampling_code == "smote_tomek":
        return SMOTETomek(
            random_state=RANDOM_STATE,
            smote=SMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=k_neighbors,
            ),
        )

    if resampling_code == "smote_enn":
        return SMOTEENN(
            random_state=RANDOM_STATE,
            smote=SMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=k_neighbors,
            ),
        )

    if resampling_code == "none":
        return None

    raise ValueError(
        f"Unknown resampling strategy: "
        f"{resampling_code}"
    )


def build_pipeline(
    resampling_code,
    classifier_name,
    numeric_features,
    categorical_features,
    y,
):
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    classifier = build_classifier(
        classifier_name
    )

    steps = [
        (
            "preprocessor",
            preprocessor,
        )
    ]

    if resampling_code != "none":
        resampler = build_resampler(
            resampling_code,
            y,
        )

        steps.append(
            (
                "resampler",
                resampler,
            )
        )

    steps.append(
        (
            "classifier",
            classifier,
        )
    )

    return ImbPipeline(
        steps=steps
    )


def calculate_macro_f1_mathematically(cm):
    if cm.shape != (2, 2):
        raise ValueError(
            "Expected a 2x2 confusion matrix."
        )

    tn, fp, fn, tp = cm.ravel()

    def calculate_f1(
        tp_value,
        fp_value,
        fn_value,
    ):
        precision_denominator = (
            tp_value + fp_value
        )

        recall_denominator = (
            tp_value + fn_value
        )

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


def run_classifier_benchmark(
    dataset_name,
    X,
    y,
    numeric_features,
    categorical_features,
    classifier_name,
):
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    safe_k = get_safe_k_neighbors(y)

    print("\n")
    print("=" * 75)
    print(
        f"{dataset_name} - "
        f"{classifier_name}"
    )
    print("=" * 75)

    print(
        f"\nNumeric features     : "
        f"{len(numeric_features)}"
    )

    print(
        f"Categorical features : "
        f"{len(categorical_features)}"
    )

    print(
        f"Safe neighbor k      : "
        f"{safe_k}"
    )

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
            numeric_features,
            categorical_features,
            y,
        )

        start_time = time.perf_counter()

        try:
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

            failure_reason = ""

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

        except Exception as error:
            elapsed_time = (
                time.perf_counter()
                - start_time
            )

            if strategy_code != "none":
                verification = "NOT_APPLICABLE"
                failure_reason = str(error)

                tn = float("nan")
                fp = float("nan")
                fn = float("nan")
                tp = float("nan")

                accuracy = float("nan")
                balanced_accuracy = float("nan")
                precision = float("nan")
                recall = float("nan")

                f1_class_0_math = float("nan")
                f1_class_1_math = float("nan")
                macro_f1_code = float("nan")
                macro_f1_math = float("nan")
                metric_error = float("nan")

                print(
                    "Status            : "
                    "NOT_APPLICABLE"
                )

                print(
                    f"Reason            : "
                    f"{failure_reason}"
                )

                print(
                    f"Execution Time    : "
                    f"{elapsed_time:.6f} seconds"
                )

            else:
                raise

        results.append(
            {
                "dataset": dataset_name,
                "resampling_code": strategy_code,
                "resampling": strategy_name,
                "classifier": classifier_name,
                "samples": X.shape[0],
                "features": X.shape[1],
                "numeric_features": len(
                    numeric_features
                ),
                "categorical_features": len(
                    categorical_features
                ),
                "safe_k_neighbors": safe_k,
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
                "failure_reason": failure_reason,
                "execution_time_seconds": elapsed_time,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        by="macro_f1_math",
        ascending=False,
        na_position="last",
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
            "pima, zoo-3, or abalone9-18."
        ),
    )

    args = parser.parse_args()

    dataset_name = args.dataset

    (
        X,
        y,
        majority_class,
        minority_class,
        numeric_features,
        categorical_features,
    ) = load_dataset(dataset_name)

    print("=" * 75)
    print(
        "ResampleRank - Full Dataset Benchmark"
    )
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
        f"Numeric features  : "
        f"{len(numeric_features)}"
    )

    print(
        f"Categorical       : "
        f"{len(categorical_features)}"
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
            numeric_features,
            categorical_features,
            classifier_name,
        )

        all_results.append(
            result_df
        )

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

    successful_experiments = (
        combined["verification"] == "PASS"
    ).sum()

    verification_failures = (
        combined["verification"] == "FAIL"
    ).sum()

    not_applicable = (
        combined["verification"]
        == "NOT_APPLICABLE"
    ).sum()

    valid_macro_f1 = combined.loc[
        combined["verification"] == "PASS",
        "macro_f1_math",
    ]

    print("\n")
    print("=" * 75)
    print("BENCHMARK SUMMARY")
    print("=" * 75)

    print(
        f"Total experiments       : "
        f"{len(combined)}"
    )

    print(
        f"Successful experiments  : "
        f"{successful_experiments}"
    )

    print(
        f"Verification failures  : "
        f"{verification_failures}"
    )

    print(
        f"Not applicable          : "
        f"{not_applicable}"
    )

    if not valid_macro_f1.empty:
        print(
            f"Macro-F1 minimum       : "
            f"{valid_macro_f1.min():.6f}"
        )

        print(
            f"Macro-F1 maximum       : "
            f"{valid_macro_f1.max():.6f}"
        )
    else:
        print(
            "Macro-F1 minimum       : N/A"
        )

        print(
            "Macro-F1 maximum       : N/A"
        )

    print(
        "\nCombined results saved to:"
    )

    print(combined_file)


if __name__ == "__main__":
    main()