from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd

from scipy.stats import spearmanr, kendalltau

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]

META_DATASET_FILE = (
    BASE_DIR
    / "results"
    / "meta_dataset.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "meta_model"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


RANDOM_STATE = 42


NUMERIC_FEATURES = [
    "samples",
    "features",
    "feature_sample_ratio",
    "numeric_features",
    "categorical_features",
    "missing_values",
    "duplicates",
    "majority_count",
    "minority_count",
    "minority_proportion",
    "imbalance_ratio",
]


CATEGORICAL_FEATURES = [
    "resampling_code",
]


TARGET = "target_macro_f1"
GROUP = "dataset"

CLASSIFIER_COLUMN = "classifier"


CLASSIFIERS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
]


META_MODELS = {
    "ridge": Ridge(
        alpha=1.0
    ),
    "random_forest": RandomForestRegressor(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),
    "gradient_boosting": GradientBoostingRegressor(
        random_state=RANDOM_STATE,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
    ),
}


def load_meta_dataset():
    if not META_DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Meta-dataset not found: "
            f"{META_DATASET_FILE}"
        )

    df = pd.read_csv(
        META_DATASET_FILE
    )

    if df.empty:
        raise ValueError(
            "Meta-dataset is empty."
        )

    required_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [
            TARGET,
            GROUP,
            CLASSIFIER_COLUMN,
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    if df[TARGET].isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    if df[GROUP].isna().any():
        raise ValueError(
            "Dataset group contains missing values."
        )

    if df[
        CLASSIFIER_COLUMN
    ].isna().any():
        raise ValueError(
            "Classifier column contains missing values."
        )

    return df


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_pipeline(model):
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "model",
                clone(model),
            ),
        ]
    )


def get_prediction_order(group):
    """
    Return rows ordered by predicted Macro-F1 descending.

    Stable sorting is used so ties remain deterministic.
    """

    return group.sort_values(
        by=[
            "predicted_macro_f1",
            "resampling_code",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )


def evaluate_one_dataset_classifier(
    group,
):
    """
    Evaluate ranking for one dataset and one classifier.
    """

    actual_max = group[
        TARGET
    ].max()

    actual_best = group[
        group[TARGET] == actual_max
    ][
        "resampling_code"
    ].tolist()

    ranked = get_prediction_order(
        group
    )

    predicted_best = ranked.iloc[0][
        "resampling_code"
    ]

    actual_best_set = set(
        actual_best
    )

    top1_correct = (
        predicted_best
        in actual_best_set
    )

    top3_predictions = set(
        ranked.head(3)[
            "resampling_code"
        ]
    )

    top3_correct = bool(
        top3_predictions
        & actual_best_set
    )

    predicted_choice_actual_score = (
        float(
            ranked.iloc[0][TARGET]
        )
    )

    actual_best_score = float(
        actual_max
    )

    regret = (
        actual_best_score
        - predicted_choice_actual_score
    )

    actual_values = group[
        TARGET
    ].to_numpy()

    predicted_values = group[
        "predicted_macro_f1"
    ].to_numpy()

    spearman_value = np.nan
    kendall_value = np.nan

    try:
        spearman_result = spearmanr(
            actual_values,
            predicted_values,
        )

        spearman_value = float(
            spearman_result.statistic
        )

    except Exception:
        spearman_value = np.nan

    try:
        kendall_result = kendalltau(
            actual_values,
            predicted_values,
        )

        kendall_value = float(
            kendall_result.statistic
        )

    except Exception:
        kendall_value = np.nan

    return {
        "dataset": group.iloc[0][
            "dataset"
        ],
        "classifier": group.iloc[0][
            CLASSIFIER_COLUMN
        ],
        "number_of_resampling_strategies": len(
            group
        ),
        "actual_best_resampling": (
            "|".join(actual_best)
        ),
        "predicted_best_resampling": (
            predicted_best
        ),
        "actual_best_macro_f1": (
            actual_best_score
        ),
        "predicted_choice_actual_macro_f1": (
            predicted_choice_actual_score
        ),
        "regret": regret,
        "top1_correct": top1_correct,
        "top3_correct": top3_correct,
        "spearman": spearman_value,
        "kendall": kendall_value,
    }


def calculate_ranking_metrics(
    evaluation_df,
):
    ranking_rows = []

    for (
        dataset_name,
        classifier_name,
    ), group in evaluation_df.groupby(
        [
            GROUP,
            CLASSIFIER_COLUMN,
        ]
    ):

        ranking_result = (
            evaluate_one_dataset_classifier(
                group
            )
        )

        ranking_rows.append(
            ranking_result
        )

    ranking_df = pd.DataFrame(
        ranking_rows
    )

    if ranking_df.empty:
        raise ValueError(
            "No ranking evaluation rows were produced."
        )

    return ranking_df


def fit_leave_one_dataset_out(
    model,
    classifier_df,
):
    """
    Perform Leave-One-Dataset-Out evaluation for
    one classifier.

    Each test fold contains one complete dataset.
    """

    logo = LeaveOneGroupOut()

    X = classifier_df[
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    ]

    y = classifier_df[
        TARGET
    ]

    groups = classifier_df[
        GROUP
    ]

    predictions = np.full(
        len(classifier_df),
        np.nan,
        dtype=float,
    )

    fold_records = []

    for (
        fold_number,
        (
            train_index,
            test_index,
        ),
    ) in enumerate(
        logo.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):

        X_train = X.iloc[
            train_index
        ]

        X_test = X.iloc[
            test_index
        ]

        y_train = y.iloc[
            train_index
        ]

        test_datasets = (
            classifier_df.iloc[
                test_index
            ][GROUP]
            .unique()
            .tolist()
        )

        pipeline = build_pipeline(
            model
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        predictions[
            test_index
        ] = pipeline.predict(
            X_test
        )

        fold_records.append(
            {
                "fold": fold_number,
                "test_datasets": "|".join(
                    test_datasets
                ),
                "train_rows": len(
                    train_index
                ),
                "test_rows": len(
                    test_index
                ),
            }
        )

    if np.isnan(
        predictions
    ).any():

        raise ValueError(
            "LODO predictions contain missing values."
        )

    return predictions, fold_records


def evaluate_single_model(
    model_name,
    model,
    df,
):
    """
    Evaluate one meta-model independently for each
    classifier.
    """

    classifier_results = []
    all_predictions = []
    all_fold_records = []

    for classifier_name in CLASSIFIERS:

        classifier_df = df[
            df[CLASSIFIER_COLUMN]
            == classifier_name
        ].copy()

        if classifier_df.empty:
            raise ValueError(
                f"No rows found for classifier: "
                f"{classifier_name}"
            )

        print("\n" + "-" * 75)
        print(
            f"{model_name} | "
            f"{classifier_name}"
        )
        print("-" * 75)

        unique_datasets = (
            classifier_df[
                GROUP
            ].nunique()
        )

        print(
            f"Datasets          : "
            f"{unique_datasets}"
        )

        print(
            f"Experiment rows   : "
            f"{len(classifier_df)}"
        )

        predictions, fold_records = (
            fit_leave_one_dataset_out(
                model,
                classifier_df,
            )
        )

        classifier_df[
            "predicted_macro_f1"
        ] = predictions

        ranking_df = (
            calculate_ranking_metrics(
                classifier_df
            )
        )

        y_true = classifier_df[
            TARGET
        ]

        y_pred = classifier_df[
            "predicted_macro_f1"
        ]

        mae = mean_absolute_error(
            y_true,
            y_pred,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )

        top1_accuracy = (
            ranking_df[
                "top1_correct"
            ].mean()
        )

        top3_accuracy = (
            ranking_df[
                "top3_correct"
            ].mean()
        )

        spearman_valid = (
            ranking_df[
                "spearman"
            ].dropna()
        )

        kendall_valid = (
            ranking_df[
                "kendall"
            ].dropna()
        )

        mean_spearman = (
            spearman_valid.mean()
            if not spearman_valid.empty
            else np.nan
        )

        mean_kendall = (
            kendall_valid.mean()
            if not kendall_valid.empty
            else np.nan
        )

        mean_regret = (
            ranking_df[
                "regret"
            ].mean()
        )

        print(
            f"MAE                 : "
            f"{mae:.6f}"
        )

        print(
            f"RMSE                : "
            f"{rmse:.6f}"
        )

        print(
            f"Top-1 resampling    : "
            f"{top1_accuracy:.6f}"
        )

        print(
            f"Top-3 resampling    : "
            f"{top3_accuracy:.6f}"
        )

        print(
            f"Mean Spearman       : "
            f"{mean_spearman:.6f}"
            if not np.isnan(
                mean_spearman
            )
            else
            "Mean Spearman       : N/A"
        )

        print(
            f"Mean Kendall        : "
            f"{mean_kendall:.6f}"
            if not np.isnan(
                mean_kendall
            )
            else
            "Mean Kendall        : N/A"
        )

        print(
            f"Mean Regret         : "
            f"{mean_regret:.6f}"
        )

        print(
            f"Valid Spearman sets : "
            f"{len(spearman_valid)} / "
            f"{len(ranking_df)}"
        )

        print(
            f"Valid Kendall sets  : "
            f"{len(kendall_valid)} / "
            f"{len(ranking_df)}"
        )

        ranking_output = (
            ranking_df.copy()
        )

        ranking_output[
            "meta_model"
        ] = model_name

        all_predictions.append(
            classifier_df
        )

        ranking_file = (
            OUTPUT_DIR
            / f"{model_name}_{classifier_name}_ranking.csv"
        )

        ranking_output.to_csv(
            ranking_file,
            index=False,
        )

        prediction_file = (
            OUTPUT_DIR
            / f"{model_name}_{classifier_name}_predictions.csv"
        )

        classifier_df.to_csv(
            prediction_file,
            index=False,
        )

        for record in fold_records:
            record[
                "model"
            ] = model_name

            record[
                "classifier"
            ] = classifier_name

        all_fold_records.extend(
            fold_records
        )

        classifier_results.append(
            {
                "model": model_name,
                "classifier": classifier_name,
                "rows": len(
                    classifier_df
                ),
                "datasets": unique_datasets,
                "mae": mae,
                "rmse": rmse,
                "top1_resampling_accuracy": (
                    top1_accuracy
                ),
                "top3_resampling_accuracy": (
                    top3_accuracy
                ),
                "mean_spearman": (
                    mean_spearman
                ),
                "mean_kendall": (
                    mean_kendall
                ),
                "mean_regret": (
                    mean_regret
                ),
                "valid_spearman_sets": (
                    len(spearman_valid)
                ),
                "valid_kendall_sets": (
                    len(kendall_valid)
                ),
            }
        )

    model_results_df = pd.DataFrame(
        classifier_results
    )

    combined_predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    combined_ranking = (
        calculate_ranking_metrics(
            combined_predictions
        )
    )

    overall_true = (
        combined_predictions[TARGET]
    )

    overall_pred = (
        combined_predictions[
            "predicted_macro_f1"
        ]
    )

    overall_mae = mean_absolute_error(
        overall_true,
        overall_pred,
    )

    overall_rmse = np.sqrt(
        mean_squared_error(
            overall_true,
            overall_pred,
        )
    )

    overall_top1 = (
        combined_ranking[
            "top1_correct"
        ].mean()
    )

    overall_top3 = (
        combined_ranking[
            "top3_correct"
        ].mean()
    )

    overall_spearman = (
        combined_ranking[
            "spearman"
        ].dropna().mean()
    )

    overall_kendall = (
        combined_ranking[
            "kendall"
        ].dropna().mean()
    )

    overall_regret = (
        combined_ranking[
            "regret"
        ].mean()
    )

    overall_summary = {
        "model": model_name,
        "classifier": "ALL",
        "rows": len(
            combined_predictions
        ),
        "datasets": combined_ranking[
            GROUP
        ].nunique(),
        "mae": overall_mae,
        "rmse": overall_rmse,
        "top1_resampling_accuracy": (
            overall_top1
        ),
        "top3_resampling_accuracy": (
            overall_top3
        ),
        "mean_spearman": (
            overall_spearman
        ),
        "mean_kendall": (
            overall_kendall
        ),
        "mean_regret": (
            overall_regret
        ),
        "valid_spearman_sets": (
            combined_ranking[
                "spearman"
            ].notna().sum()
        ),
        "valid_kendall_sets": (
            combined_ranking[
                "kendall"
            ].notna().sum()
        ),
    }

    model_results_df = pd.concat(
        [
            model_results_df,
            pd.DataFrame(
                [overall_summary]
            ),
        ],
        ignore_index=True,
    )

    folds_file = (
        OUTPUT_DIR
        / f"{model_name}_lodo_folds.csv"
    )

    pd.DataFrame(
        all_fold_records
    ).to_csv(
        folds_file,
        index=False,
    )

    return (
        model_results_df,
        combined_predictions,
        combined_ranking,
    )


def main():

    print("=" * 75)
    print(
        "ResampleRank - Resampling Meta-Learning Evaluation"
    )
    print("=" * 75)

    df = load_meta_dataset()

    print(
        f"\nMeta-dataset shape : "
        f"{df.shape}"
    )

    print(
        f"Datasets           : "
        f"{df[GROUP].nunique()}"
    )

    print(
        f"Model-ready rows   : "
        f"{len(df)}"
    )

    print("\nEvaluation strategy:")
    print(
        "Leave-One-Dataset-Out "
        "Cross-Validation"
    )

    print(
        "\nMeta-learning task:"
    )

    print(
        "For each classifier, predict and rank "
        "the resampling strategies on an unseen dataset."
    )

    print(
        "\nNumeric meta-features:"
    )

    print(
        NUMERIC_FEATURES
    )

    print(
        "\nCategorical meta-feature:"
    )

    print(
        CATEGORICAL_FEATURES
    )

    print(
        f"\nTarget: {TARGET}"
    )

    print(
        "\nClassifiers evaluated:"
    )

    print(
        CLASSIFIERS
    )

    all_model_results = []

    for (
        model_name,
        model,
    ) in META_MODELS.items():

        (
            model_results_df,
            _,
            _,
        ) = evaluate_single_model(
            model_name,
            model,
            df,
        )

        all_model_results.append(
            model_results_df
        )

    summary_df = pd.concat(
        all_model_results,
        ignore_index=True,
    )

    summary_file = (
        OUTPUT_DIR
        / "meta_model_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    print("\n" + "=" * 75)
    print(
        "META-MODEL SUMMARY"
    )
    print("=" * 75)

    print(
        summary_df.to_string(
            index=False
        )
    )

    overall_df = summary_df[
        summary_df[
            "classifier"
        ] == "ALL"
    ].copy()

    overall_file = (
        OUTPUT_DIR
        / "meta_model_overall.csv"
    )

    overall_df.to_csv(
        overall_file,
        index=False,
    )

    print("\n" + "-" * 75)
    print(
        "OVERALL META-MODEL RESULTS"
    )
    print("-" * 75)

    print(
        overall_df.to_string(
            index=False
        )
    )

    print("\nSaved files:")
    print(summary_file)
    print(overall_file)


if __name__ == "__main__":
    main()