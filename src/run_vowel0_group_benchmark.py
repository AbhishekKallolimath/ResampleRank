from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from xgboost import XGBClassifier

from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    ADASYN,
)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = (
    BASE_DIR
    / "data"
    / "raw"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "sensitivity"
    / "vowel0_group"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DATASET_NAME = "vowel0"
GROUP_COLUMN = "SpeakerNumber"
RANDOM_STATE = 42
N_SPLITS = 5


RESAMPLING_STRATEGIES = [
    ("none", "No Resampling"),
    ("ros", "Random Oversampling"),
    ("rus", "Random Undersampling"),
    ("smote", "SMOTE"),
    ("adasyn", "ADASYN"),
    ("smote_tomek", "SMOTE-Tomek"),
    ("smote_enn", "SMOTE-ENN"),
]


CLASSIFIERS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
]


def load_dataset():
    feature_file = (
        RAW_DIR
        / f"{DATASET_NAME}.csv"
    )

    target_file = (
        RAW_DIR
        / f"{DATASET_NAME}_target.csv"
    )

    if not feature_file.exists():
        raise FileNotFoundError(
            f"Feature file not found: {feature_file}"
        )

    if not target_file.exists():
        raise FileNotFoundError(
            f"Target file not found: {target_file}"
        )

    X = pd.read_csv(
        feature_file
    )

    y_raw = (
        pd.read_csv(
            target_file
        )
        .iloc[:, 0]
        .astype(str)
        .str.strip()
    )

    if GROUP_COLUMN not in X.columns:
        raise ValueError(
            f"{GROUP_COLUMN} not found in Vowel0 features."
        )

    class_counts = (
        y_raw.value_counts()
    )

    if len(class_counts) != 2:
        raise ValueError(
            "Vowel0 must contain exactly two classes."
        )

    majority_class = (
        class_counts.idxmax()
    )

    minority_class = (
        class_counts.idxmin()
    )

    y = (
        y_raw
        .map(
            {
                majority_class: 0,
                minority_class: 1,
            }
        )
        .astype(int)
    )

    groups = (
        X[GROUP_COLUMN]
        .copy()
    )

    X_model = X.drop(
        columns=[GROUP_COLUMN]
    ).copy()

    return (
        X_model,
        y,
        groups,
        majority_class,
        minority_class,
    )


def build_classifier(
    classifier_name,
):
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


def get_safe_k_neighbors(
    y_train,
):
    counts = (
        pd.Series(y_train)
        .value_counts()
    )

    if len(counts) != 2:
        raise ValueError(
            "Training fold does not contain both classes."
        )

    minority_count = (
        counts.min()
    )

    if minority_count < 2:
        raise ValueError(
            "Training fold has fewer than 2 minority samples."
        )

    return int(
        min(
            5,
            minority_count - 1,
        )
    )


def build_pipeline(
    resampling_code,
    classifier_name,
    k_neighbors,
):
    classifier = build_classifier(
        classifier_name
    )

    if resampling_code == "none":
        return SklearnPipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    classifier,
                ),
            ]
        )

    if resampling_code == "ros":
        sampler = RandomOverSampler(
            random_state=RANDOM_STATE
        )

    elif resampling_code == "rus":
        sampler = RandomUnderSampler(
            random_state=RANDOM_STATE
        )

    elif resampling_code == "smote":
        sampler = SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=k_neighbors,
        )

    elif resampling_code == "adasyn":
        sampler = ADASYN(
            random_state=RANDOM_STATE,
            n_neighbors=k_neighbors,
        )

    elif resampling_code == "smote_tomek":
        sampler = SMOTETomek(
            random_state=RANDOM_STATE,
            smote=SMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=k_neighbors,
            ),
        )

    elif resampling_code == "smote_enn":
        sampler = SMOTEENN(
            random_state=RANDOM_STATE,
            smote=SMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=k_neighbors,
            ),
        )

    else:
        raise ValueError(
            f"Unknown resampling strategy: {resampling_code}"
        )

    return ImbPipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "resampler",
                sampler,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def calculate_macro_f1_math(
    cm,
):
    tn, fp, fn, tp = (
        cm.ravel()
    )

    precision_0 = (
        tn / (tn + fn)
        if (tn + fn) != 0
        else 0.0
    )

    recall_0 = (
        tn / (tn + fp)
        if (tn + fp) != 0
        else 0.0
    )

    if (
        precision_0
        + recall_0
        == 0
    ):
        f1_0 = 0.0
    else:
        f1_0 = (
            2
            * precision_0
            * recall_0
            / (
                precision_0
                + recall_0
            )
        )

    precision_1 = (
        tp / (tp + fp)
        if (tp + fp) != 0
        else 0.0
    )

    recall_1 = (
        tp / (tp + fn)
        if (tp + fn) != 0
        else 0.0
    )

    if (
        precision_1
        + recall_1
        == 0
    ):
        f1_1 = 0.0
    else:
        f1_1 = (
            2
            * precision_1
            * recall_1
            / (
                precision_1
                + recall_1
            )
        )

    return (
        f1_0
        + f1_1
    ) / 2.0


def run_classifier(
    X,
    y,
    groups,
    classifier_name,
):
    cv = GroupKFold(
        n_splits=N_SPLITS
    )

    splits = list(
        cv.split(
            X,
            y,
            groups,
        )
    )

    minimum_training_minority = min(
        (
            pd.Series(
                y.iloc[train_index]
            )
            .value_counts()
            .min()
            for train_index, _ in splits
        )
    )

    if minimum_training_minority < 2:
        raise ValueError(
            "A training fold contains fewer than 2 minority samples."
        )

    k_neighbors = min(
        5,
        int(
            minimum_training_minority
            - 1
        ),
    )

    results = []

    for (
        strategy_code,
        strategy_name,
    ) in RESAMPLING_STRATEGIES:

        print(
            f"Running: "
            f"{strategy_name} | "
            f"{classifier_name}"
        )

        model = build_pipeline(
            strategy_code,
            classifier_name,
            k_neighbors,
        )

        start_time = time.perf_counter()

        y_pred = cross_val_predict(
            model,
            X,
            y,
            cv=splits,
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

        tn, fp, fn, tp = (
            cm.ravel()
        )

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

        macro_f1_math = (
            calculate_macro_f1_math(
                cm
            )
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

        results.append(
            {
                "dataset": DATASET_NAME,
                "validation": "GroupKFold",
                "group_column": GROUP_COLUMN,
                "excluded_from_features": GROUP_COLUMN,
                "groups": groups.nunique(),
                "resampling_code": strategy_code,
                "resampling": strategy_name,
                "classifier": classifier_name,
                "samples": X.shape[0],
                "features_used": X.shape[1],
                "safe_k_neighbors": k_neighbors,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "precision": precision,
                "recall": recall,
                "macro_f1_code": macro_f1_code,
                "macro_f1_math": macro_f1_math,
                "metric_error": metric_error,
                "verification": verification,
                "execution_time_seconds": elapsed_time,
            }
        )

        print(
            f"  Macro-F1: "
            f"{macro_f1_math:.6f}"
        )

    return pd.DataFrame(
        results
    )


def main():

    (
        X,
        y,
        groups,
        majority_class,
        minority_class,
    ) = load_dataset()

    print(
        "=" * 75
    )

    print(
        "ResampleRank - Vowel0 Speaker-Aware Benchmark"
    )

    print(
        "=" * 75
    )

    print(
        f"Dataset          : {DATASET_NAME}"
    )

    print(
        f"Samples          : {len(X)}"
    )

    print(
        f"Original features: {X.shape[1] + 1}"
    )

    print(
        f"Features used    : {X.shape[1]}"
    )

    print(
        f"Excluded feature : {GROUP_COLUMN}"
    )

    print(
        f"Groups           : {groups.nunique()}"
    )

    print(
        f"Validation       : GroupKFold({N_SPLITS})"
    )

    print(
        f"Majority class   : {majority_class}"
    )

    print(
        f"Minority class   : {minority_class}"
    )

    all_results = []

    for classifier_name in CLASSIFIERS:

        classifier_results = run_classifier(
            X,
            y,
            groups,
            classifier_name,
        )

        result_file = (
            OUTPUT_DIR
            / f"vowel0_{classifier_name}_group_results.csv"
        )

        classifier_results.to_csv(
            result_file,
            index=False,
        )

        print(
            f"\nSaved: {result_file}"
        )

        all_results.append(
            classifier_results
        )

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    combined_file = (
        OUTPUT_DIR
        / "vowel0_all_group_results.csv"
    )

    combined.to_csv(
        combined_file,
        index=False,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "BENCHMARK SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"Total experiment slots : {len(combined)}"
    )

    print(
        "Verification failures  : "
        f"{(combined['verification'] != 'PASS').sum()}"
    )

    print(
        f"Macro-F1 minimum       : "
        f"{combined['macro_f1_math'].min():.6f}"
    )

    print(
        f"Macro-F1 maximum       : "
        f"{combined['macro_f1_math'].max():.6f}"
    )

    print(
        f"\nCombined results saved to:"
    )

    print(
        combined_file
    )


if __name__ == "__main__":
    main()