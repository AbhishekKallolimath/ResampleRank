from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
)

from xgboost import XGBClassifier

from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    ADASYN,
)
from imblearn.under_sampling import (
    RandomUnderSampler,
)
from imblearn.combine import (
    SMOTETomek,
    SMOTEENN,
)
from imblearn.pipeline import Pipeline as ImbPipeline


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = (
    BASE_DIR
    / "data"
    / "raw"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "sensitivity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DATASETS = [
    "wisconsin",
    "vowel0",
]


CLASSIFIERS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
]


RESAMPLING_STRATEGIES = [
    ("none", "No Resampling"),
    ("ros", "Random Oversampling"),
    ("rus", "Random Undersampling"),
    ("smote", "SMOTE"),
    ("adasyn", "ADASYN"),
    ("smote_tomek", "SMOTE-Tomek"),
    ("smote_enn", "SMOTE-ENN"),
]


RANDOM_STATE = 42


def load_dataset(
    dataset_name,
):
    feature_file = (
        RAW_DIR
        / f"{dataset_name}.csv"
    )

    target_file = (
        RAW_DIR
        / f"{dataset_name}_target.csv"
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

    y_df = pd.read_csv(
        target_file
    )

    if y_df.shape[1] != 1:
        raise ValueError(
            f"Expected one target column, "
            f"found {y_df.shape[1]}"
        )

    y_raw = (
        y_df.iloc[:, 0]
        .astype(str)
        .str.strip()
    )

    class_counts = (
        y_raw.value_counts()
    )

    if len(class_counts) != 2:
        raise ValueError(
            f"Expected binary classification, "
            f"found {len(class_counts)} classes"
        )

    majority_class = (
        class_counts.idxmax()
    )

    minority_class = (
        class_counts.idxmin()
    )

    label_mapping = {
        majority_class: 0,
        minority_class: 1,
    }

    y = (
        y_raw
        .map(label_mapping)
        .astype(int)
    )

    return (
        X,
        y,
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
    minority_count = (
        pd.Series(y_train)
        .value_counts()
        .min()
    )

    if minority_count < 2:
        raise ValueError(
            "Minority class has fewer than 2 "
            "training samples."
        )

    return int(
        max(
            1,
            min(
                5,
                minority_count - 1,
            ),
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

        return Pipeline(
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
            f"Unknown resampling strategy: "
            f"{resampling_code}"
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


def build_groups(
    dataset_name,
    X,
):
    if dataset_name == "wisconsin":

        duplicate_keys = (
            X.astype(str)
            .fillna("<NA>")
            .astype(str)
            .agg(
                "||".join,
                axis=1,
            )
        )

        group_codes, _ = pd.factorize(
            duplicate_keys,
            sort=False,
        )

        return (
            pd.Series(
                group_codes,
                index=X.index,
            ),
            "duplicate_group",
        )

    if dataset_name == "vowel0":

        if "SpeakerNumber" not in X.columns:
            raise ValueError(
                "SpeakerNumber column not found "
                "in vowel0 dataset."
            )

        return (
            X["SpeakerNumber"]
            .copy(),
            "SpeakerNumber",
        )

    raise ValueError(
        f"No sensitivity grouping defined "
        f"for {dataset_name}"
    )


def calculate_macro_f1_from_confusion_matrix(
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


def run_single_experiment(
    dataset_name,
    X,
    y,
    groups,
    grouping_type,
    resampling_code,
    resampling_name,
    classifier_name,
):
    fold_count = 5

    group_cv = GroupKFold(
        n_splits=fold_count
    )

    split_folds = list(
        group_cv.split(
            X,
            y,
            groups,
        )
    )

    minimum_training_minority = (
        min(
            pd.Series(
                y.iloc[train_index]
            )
            .value_counts()
            .min()
            for (
                train_index,
                _
            ) in split_folds
        )
    )

    if minimum_training_minority < 2:
        raise ValueError(
            "At least one training fold has "
            "fewer than 2 minority samples."
        )

    k_neighbors = (
        get_safe_k_neighbors(
            y.iloc[
                split_folds[0][0]
            ]
        )
    )

    k_neighbors = min(
        k_neighbors,
        int(
            minimum_training_minority - 1
        ),
    )

    pipeline = build_pipeline(
        resampling_code,
        classifier_name,
        k_neighbors,
    )

    start_time = time.perf_counter()

    y_pred = cross_val_predict(
        pipeline,
        X,
        y,
        cv=split_folds,
        method="predict",
        n_jobs=None,
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

    macro_f1_sklearn = (
        f1_score(
            y,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    macro_f1_math = (
        calculate_macro_f1_from_confusion_matrix(
            cm
        )
    )

    metric_error = abs(
        macro_f1_math
        - macro_f1_sklearn
    )

    return {
        "dataset": dataset_name,
        "sensitivity_type": grouping_type,
        "resampling_code": resampling_code,
        "resampling": resampling_name,
        "classifier": classifier_name,
        "samples": X.shape[0],
        "features": X.shape[1],
        "groups": pd.Series(
            groups
        ).nunique(),
        "safe_k_neighbors": k_neighbors,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "macro_f1_sensitivity": macro_f1_math,
        "macro_f1_sklearn": macro_f1_sklearn,
        "metric_error": metric_error,
        "execution_time_seconds": elapsed_time,
        "verification": (
            "PASS"
            if metric_error < 1e-6
            else "FAIL"
        ),
    }


def main():

    all_results = []

    print(
        "=" * 75
    )

    print(
        "ResampleRank - Sensitivity Analysis"
    )

    print(
        "=" * 75
    )

    for dataset_name in DATASETS:

        print(
            f"\n{'-' * 75}"
        )

        print(
            f"Dataset: {dataset_name}"
        )

        print(
            f"{'-' * 75}"
        )

        (
            X,
            y,
            majority_class,
            minority_class,
        ) = load_dataset(
            dataset_name
        )

        (
            groups,
            grouping_type,
        ) = build_groups(
            dataset_name,
            X,
        )

        print(
            f"Samples       : {len(X)}"
        )

        print(
            f"Features      : {X.shape[1]}"
        )

        print(
            f"Grouping      : {grouping_type}"
        )

        print(
            f"Groups        : {groups.nunique()}"
        )

        print(
            f"Majority      : "
            f"{majority_class}"
        )

        print(
            f"Minority      : "
            f"{minority_class}"
        )

        for (
            resampling_code,
            resampling_name,
        ) in RESAMPLING_STRATEGIES:

            for classifier_name in CLASSIFIERS:

                print(
                    f"Running "
                    f"{resampling_name} | "
                    f"{classifier_name}"
                )

                try:

                    result = (
                        run_single_experiment(
                            dataset_name,
                            X,
                            y,
                            groups,
                            grouping_type,
                            resampling_code,
                            resampling_name,
                            classifier_name,
                        )
                    )

                    all_results.append(
                        result
                    )

                    print(
                        "  Macro-F1: "
                        f"{result['macro_f1_sensitivity']:.6f}"
                    )

                except Exception as exc:

                    print(
                        "  NOT_APPLICABLE: "
                        f"{exc}"
                    )

                    all_results.append(
                        {
                            "dataset": dataset_name,
                            "sensitivity_type": grouping_type,
                            "resampling_code": resampling_code,
                            "resampling": resampling_name,
                            "classifier": classifier_name,
                            "samples": X.shape[0],
                            "features": X.shape[1],
                            "groups": groups.nunique(),
                            "safe_k_neighbors": np.nan,
                            "tn": np.nan,
                            "fp": np.nan,
                            "fn": np.nan,
                            "tp": np.nan,
                            "macro_f1_sensitivity": np.nan,
                            "macro_f1_sklearn": np.nan,
                            "metric_error": np.nan,
                            "execution_time_seconds": np.nan,
                            "verification": "NOT_APPLICABLE",
                            "failure_reason": str(exc),
                        }
                    )

    sensitivity_df = pd.DataFrame(
        all_results
    )

    original_results = []

    for dataset_name in DATASETS:

        for classifier_name in CLASSIFIERS:

            file_path = (
                RESULTS_DIR
                / f"{dataset_name}_{classifier_name}_results.csv"
            )

            if not file_path.exists():
                continue

            original_df = pd.read_csv(
                file_path
            )

            original_df = (
                original_df[
                    original_df[
                        "verification"
                    ] == "PASS"
                ][
                    [
                        "resampling_code",
                        "macro_f1_math",
                    ]
                ]
                .rename(
                    columns={
                        "macro_f1_math":
                            "macro_f1_original"
                    }
                )
            )

            original_df[
                "dataset"
            ] = dataset_name

            original_df[
                "classifier"
            ] = classifier_name

            original_results.append(
                original_df
            )

    original_df = pd.concat(
        original_results,
        ignore_index=True,
    )

    comparison_df = (
        sensitivity_df
        .merge(
            original_df,
            on=[
                "dataset",
                "classifier",
                "resampling_code",
            ],
            how="left",
        )
    )

    comparison_df[
        "macro_f1_change"
    ] = (
        comparison_df[
            "macro_f1_sensitivity"
        ]
        - comparison_df[
            "macro_f1_original"
        ]
    )

    summary_df = (
        comparison_df[
            comparison_df[
                "verification"
            ] == "PASS"
        ]
        .groupby(
            [
                "dataset",
                "sensitivity_type",
                "classifier",
            ]
        )
        .agg(
            strategies_tested=(
                "resampling_code",
                "count",
            ),
            mean_original_macro_f1=(
                "macro_f1_original",
                "mean",
            ),
            mean_sensitivity_macro_f1=(
                "macro_f1_sensitivity",
                "mean",
            ),
            mean_macro_f1_change=(
                "macro_f1_change",
                "mean",
            ),
            max_absolute_change=(
                "macro_f1_change",
                lambda x: x.abs().max(),
            ),
        )
        .reset_index()
    )

    detailed_file = (
        OUTPUT_DIR
        / "sensitivity_details.csv"
    )

    comparison_file = (
        OUTPUT_DIR
        / "sensitivity_comparison.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / "sensitivity_summary.csv"
    )

    sensitivity_df.to_csv(
        detailed_file,
        index=False,
    )

    comparison_df.to_csv(
        comparison_file,
        index=False,
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "SENSITIVITY SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved files:"
    )

    print(
        detailed_file
    )

    print(
        comparison_file
    )

    print(
        summary_file
    )


if __name__ == "__main__":
    main()