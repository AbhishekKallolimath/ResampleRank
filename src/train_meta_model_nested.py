from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import spearmanr, kendalltau
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor


RESULTS_DIR = Path("results")
META_DATASET_PATH = RESULTS_DIR / "meta_dataset.csv"
OUTPUT_DIR = RESULTS_DIR / "nested_meta_model"

DATASETS = [
    "pima",
    "glass1",
    "wisconsin",
    "yeast1",
    "haberman",
    "vehicle2",
    "iris0",
    "segment0",
    "page-blocks0",
    "vowel0",
    "shuttle-c0-vs-c4",
    "ecoli-0_vs_1",
]

CLASSIFIERS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
]

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
    "resampling_code"
]

TARGET = "delta_macro_f1_vs_baseline"

STRATEGIES = [
    "No Resampling",
    "Random Oversampling",
    "Random Undersampling",
    "SMOTE",
    "ADASYN",
    "SMOTE-Tomek",
    "SMOTE-ENN",
]

RANDOM_STATE = 42


def build_models():
    return {
        "ridge": Ridge(alpha=1.0),

        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "gradient_boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2
        ),

        "xgboost": XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="reg:squarederror"
        )
    }


def build_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse=False
                ),
                CATEGORICAL_FEATURES
            )
        ],
        remainder="drop"
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )


def load_meta_dataset():
    if not META_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Meta-dataset not found: {META_DATASET_PATH}"
        )

    df = pd.read_csv(
        META_DATASET_PATH
    )

    required_columns = (
        [
            "dataset",
            "classifier",
            "resampling",
            "resampling_code",
            TARGET,
            "target_macro_f1",
            "baseline_macro_f1"
        ]
        + NUMERIC_FEATURES
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    return df.copy()


def create_dataset_splits():
    rng = np.random.default_rng(
        RANDOM_STATE
    )

    shuffled = list(DATASETS)

    rng.shuffle(
        shuffled
    )

    rows = []

    n = len(shuffled)

    for i, test_dataset in enumerate(
        shuffled
    ):
        validation_1 = shuffled[
            (i + 1) % n
        ]

        validation_2 = shuffled[
            (i + 2) % n
        ]

        validation = [
            validation_1,
            validation_2
        ]

        train = [
            dataset
            for dataset in shuffled
            if dataset != test_dataset
            and dataset not in validation
        ]

        rows.append({
            "fold": i + 1,
            "test_dataset": test_dataset,
            "validation_dataset_1": validation_1,
            "validation_dataset_2": validation_2,
            "train_datasets": "|".join(train),
            "train_count": len(train),
            "validation_count": len(validation),
            "test_count": 1
        })

    return pd.DataFrame(rows)


def safe_correlation(values_a, values_b):
    a = np.asarray(
        values_a,
        dtype=float
    )

    b = np.asarray(
        values_b,
        dtype=float
    )

    if len(a) < 2 or len(b) < 2:
        return np.nan

    if (
        np.allclose(a, a[0])
        or np.allclose(b, b[0])
    ):
        return np.nan

    try:
        value = spearmanr(
            a,
            b
        )[0]

        if value is None:
            return np.nan

        return float(value)

    except Exception:
        return np.nan


def safe_kendall(values_a, values_b):
    a = np.asarray(
        values_a,
        dtype=float
    )

    b = np.asarray(
        values_b,
        dtype=float
    )

    if len(a) < 2 or len(b) < 2:
        return np.nan

    if (
        np.allclose(a, a[0])
        or np.allclose(b, b[0])
    ):
        return np.nan

    try:
        value = kendalltau(
            a,
            b
        )[0]

        if value is None:
            return np.nan

        return float(value)

    except Exception:
        return np.nan


def evaluate_predictions(
    test_df,
    predictions,
    fold,
    meta_model,
    classifier,
    stage
):
    temp = test_df.copy()

    temp = temp.reset_index(
        drop=True
    )

    temp[
        "predicted_delta_macro_f1"
    ] = predictions

    temp[
        "predicted_macro_f1"
    ] = (
        temp["baseline_macro_f1"]
        + temp["predicted_delta_macro_f1"]
    ).clip(
        0.0,
        1.0
    )

    actual_values = temp[
        "delta_macro_f1_vs_baseline"
    ].to_numpy()

    predicted_values = temp[
        "predicted_delta_macro_f1"
    ].to_numpy()

    mae = float(
        np.mean(
            np.abs(
                actual_values
                - predicted_values
            )
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    actual_values
                    - predicted_values
                ) ** 2
            )
        )
    )

    output_rows = []

    for dataset in temp[
        "dataset"
    ].unique():

        subset = temp[
            temp["dataset"] == dataset
        ].copy()

        actual_series = (
            subset
            .set_index("resampling")
            ["target_macro_f1"]
        )

        predicted_series = (
            subset
            .set_index("resampling")
            ["predicted_macro_f1"]
        )

        actual_series = (
            actual_series
            .reindex(STRATEGIES)
            .dropna()
        )

        predicted_series = (
            predicted_series
            .reindex(
                actual_series.index
            )
        )

        if len(actual_series) == 0:
            continue

        actual_best = (
            actual_series.idxmax()
        )

        predicted_best = (
            predicted_series.idxmax()
        )

        actual_rank = (
            predicted_series
            .rank(
                ascending=False,
                method="min"
            )
        )

        top3_strategies = (
            predicted_series
            .sort_values(
                ascending=False
            )
            .head(3)
            .index
            .tolist()
        )

        top1 = int(
            predicted_best
            == actual_best
        )

        top3 = int(
            actual_best
            in top3_strategies
        )

        selected_actual = float(
            actual_series[
                predicted_best
            ]
        )

        oracle_actual = float(
            actual_series.max()
        )

        baseline = float(
            subset[
                subset["resampling"]
                == "No Resampling"
            ]["baseline_macro_f1"]
            .iloc[0]
        )

        selected_improvement = (
            selected_actual
            - baseline
        )

        oracle_improvement = (
            oracle_actual
            - baseline
        )

        regret = (
            oracle_actual
            - selected_actual
        )

        spearman = safe_correlation(
            actual_series.values,
            predicted_series.values
        )

        kendall = safe_kendall(
            actual_series.values,
            predicted_series.values
        )

        output_rows.append({
            "fold": fold,
            "stage": stage,
            "meta_model": meta_model,
            "classifier": classifier,
            "dataset": dataset,
            "actual_best_strategy": actual_best,
            "predicted_best_strategy": predicted_best,
            "top1_accuracy": top1,
            "top3_accuracy": top3,
            "baseline_macro_f1": baseline,
            "selected_macro_f1": selected_actual,
            "oracle_macro_f1": oracle_actual,
            "selected_improvement": selected_improvement,
            "oracle_improvement": oracle_improvement,
            "regret": regret,
            "spearman": spearman,
            "kendall": kendall,
        })

    return temp, pd.DataFrame(
        output_rows
    ), mae, rmse


def summarize_selection(
    details,
    meta_model=None,
    classifier=None,
    stage=None
):
    temp = details.copy()

    if meta_model is not None:
        temp = temp[
            temp["meta_model"] == meta_model
        ]

    if classifier is not None:
        temp = temp[
            temp["classifier"] == classifier
        ]

    if stage is not None:
        temp = temp[
            temp["stage"] == stage
        ]

    if len(temp) == 0:
        return None

    valid_spearman = temp[
        "spearman"
    ].dropna()

    valid_kendall = temp[
        "kendall"
    ].dropna()

    return {
        "meta_model": (
            meta_model
            if meta_model is not None
            else "ALL"
        ),
        "classifier": (
            classifier
            if classifier is not None
            else "ALL"
        ),
        "stage": (
            stage
            if stage is not None
            else "ALL"
        ),
        "folds_or_datasets": len(temp),
        "mean_selected_improvement": temp[
            "selected_improvement"
        ].mean(),
        "mean_oracle_improvement": temp[
            "oracle_improvement"
        ].mean(),
        "mean_gap_to_oracle": temp[
            "regret"
        ].mean(),
        "mean_regret": temp[
            "regret"
        ].mean(),
        "top1_accuracy": temp[
            "top1_accuracy"
        ].mean(),
        "top3_accuracy": temp[
            "top3_accuracy"
        ].mean(),
        "mean_spearman": (
            valid_spearman.mean()
            if len(valid_spearman) > 0
            else np.nan
        ),
        "mean_kendall": (
            valid_kendall.mean()
            if len(valid_kendall) > 0
            else np.nan
        ),
    }


def main():
    print("=" * 75)
    print(
        "ResampleRank - Nested Train/Validation/Test Meta-Learning"
    )
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nLoading meta-dataset...")

    df = load_meta_dataset()

    print(
        f"Meta-dataset shape: {df.shape}"
    )

    print(
        f"Datasets: "
        f"{df['dataset'].nunique()}"
    )

    print(
        "\nProtocol:"
    )

    print(
        "9 datasets -> TRAIN"
    )

    print(
        "2 datasets -> VALIDATION"
    )

    print(
        "1 dataset  -> TEST"
    )

    split_df = create_dataset_splits()

    split_path = (
        OUTPUT_DIR /
        "nested_dataset_splits.csv"
    )

    split_df.to_csv(
        split_path,
        index=False
    )

    print(
        f"\nSaved splits: {split_path}"
    )

    validation_rows = []
    selected_model_rows = []
    test_details = []
    test_predictions = []

    for _, split in split_df.iterrows():

        fold = int(
            split["fold"]
        )

        test_dataset = (
            split["test_dataset"]
        )

        validation_datasets = [
            split["validation_dataset_1"],
            split["validation_dataset_2"]
        ]

        train_datasets = (
            split[
                "train_datasets"
            ].split("|")
        )

        print("\n" + "-" * 75)

        print(
            f"Fold {fold}"
        )

        print(
            f"Train      : {len(train_datasets)} datasets"
        )

        print(
            f"Validation : {len(validation_datasets)} datasets"
        )

        print(
            f"Test       : {test_dataset}"
        )

        for classifier in CLASSIFIERS:

            classifier_df = df[
                df["classifier"]
                == classifier
            ].copy()

            train_df = classifier_df[
                classifier_df["dataset"]
                .isin(train_datasets)
            ].copy()

            validation_df = classifier_df[
                classifier_df["dataset"]
                .isin(validation_datasets)
            ].copy()

            test_df = classifier_df[
                classifier_df["dataset"]
                == test_dataset
            ].copy()

            if len(train_df) == 0:
                continue

            if len(validation_df) == 0:
                continue

            if len(test_df) == 0:
                continue

            X_train = train_df[
                NUMERIC_FEATURES
                + CATEGORICAL_FEATURES
            ]

            y_train = train_df[
                TARGET
            ]

            X_validation = validation_df[
                NUMERIC_FEATURES
                + CATEGORICAL_FEATURES
            ]

            y_validation = validation_df[
                TARGET
            ]

            X_test = test_df[
                NUMERIC_FEATURES
                + CATEGORICAL_FEATURES
            ]

            validation_scores = []

            candidate_models = build_models()

            for model_name, model in candidate_models.items():

                pipeline = build_pipeline(
                    model
                )

                try:
                    pipeline.fit(
                        X_train,
                        y_train
                    )

                    validation_predictions = (
                        pipeline.predict(
                            X_validation
                        )
                    )

                    validation_mae = float(
                        np.mean(
                            np.abs(
                                y_validation.to_numpy()
                                - validation_predictions
                            )
                        )
                    )

                    validation_rmse = float(
                        np.sqrt(
                            np.mean(
                                (
                                    y_validation.to_numpy()
                                    - validation_predictions
                                ) ** 2
                            )
                        )
                    )

                    validation_scores.append({
                        "fold": fold,
                        "classifier": classifier,
                        "meta_model": model_name,
                        "test_dataset": test_dataset,
                        "validation_mae": validation_mae,
                        "validation_rmse": validation_rmse
                    })

                except Exception as exc:

                    validation_scores.append({
                        "fold": fold,
                        "classifier": classifier,
                        "meta_model": model_name,
                        "test_dataset": test_dataset,
                        "validation_mae": np.nan,
                        "validation_rmse": np.nan,
                        "error": str(exc)
                    })

            fold_validation_df = pd.DataFrame(
                validation_scores
            )

            validation_rows.append(
                fold_validation_df
            )

            valid_candidates = (
                fold_validation_df
                .dropna(
                    subset=[
                        "validation_mae"
                    ]
                )
                .sort_values(
                    "validation_mae"
                )
            )

            if len(valid_candidates) == 0:
                continue

            selected_model = (
                valid_candidates.iloc[0]
            )

            selected_name = (
                selected_model[
                    "meta_model"
                ]
            )

            selected_validation_mae = (
                selected_model[
                    "validation_mae"
                ]
            )

            selected_model_rows.append({
                "fold": fold,
                "classifier": classifier,
                "test_dataset": test_dataset,
                "selected_meta_model": selected_name,
                "validation_mae": selected_validation_mae,
                "validation_rmse": selected_model[
                    "validation_rmse"
                ]
            })

            print(
                f"{classifier}: "
                f"selected {selected_name} "
                f"(validation MAE = "
                f"{selected_validation_mae:.6f})"
            )

            combined_train = pd.concat(
                [
                    train_df,
                    validation_df
                ],
                ignore_index=True
            )

            X_combined = combined_train[
                NUMERIC_FEATURES
                + CATEGORICAL_FEATURES
            ]

            y_combined = combined_train[
                TARGET
            ]

            X_test = test_df[
                NUMERIC_FEATURES
                + CATEGORICAL_FEATURES
            ]

            selected_pipeline = build_pipeline(
                build_models()[
                    selected_name
                ]
            )

            try:
                selected_pipeline.fit(
                    X_combined,
                    y_combined
                )

                test_predictions_values = (
                    selected_pipeline.predict(
                        X_test
                    )
                )

                prediction_frame, selection_frame, test_mae, test_rmse = (
                    evaluate_predictions(
                        test_df,
                        test_predictions_values,
                        fold,
                        selected_name,
                        classifier,
                        "test"
                    )
                )

                test_predictions.append(
                    prediction_frame
                )

                if len(selection_frame) > 0:
                    test_details.append(
                        selection_frame
                    )

                print(
                    f"  Test MAE: "
                    f"{test_mae:.6f}"
                )

                print(
                    f"  Test RMSE: "
                    f"{test_rmse:.6f}"
                )

            except Exception as exc:

                print(
                    f"  TEST ERROR: {exc}"
                )

    if validation_rows:
        validation_result = pd.concat(
            validation_rows,
            ignore_index=True
        )
    else:
        validation_result = pd.DataFrame()

    if selected_model_rows:
        selected_models_result = pd.DataFrame(
            selected_model_rows
        )
    else:
        selected_models_result = pd.DataFrame()

    if test_predictions:
        predictions_result = pd.concat(
            test_predictions,
            ignore_index=True
        )
    else:
        predictions_result = pd.DataFrame()

    if test_details:
        details_result = pd.concat(
            test_details,
            ignore_index=True
        )
    else:
        details_result = pd.DataFrame()

    validation_path = (
        OUTPUT_DIR /
        "validation_model_comparison.csv"
    )

    selected_path = (
        OUTPUT_DIR /
        "selected_models.csv"
    )

    predictions_path = (
        OUTPUT_DIR /
        "nested_test_predictions.csv"
    )

    details_path = (
        OUTPUT_DIR /
        "nested_test_selection_results.csv"
    )

    validation_result.to_csv(
        validation_path,
        index=False
    )

    selected_models_result.to_csv(
        selected_path,
        index=False
    )

    predictions_result.to_csv(
        predictions_path,
        index=False
    )

    details_result.to_csv(
        details_path,
        index=False
    )

    summary_rows = []

    if len(details_result) > 0:

        for classifier in CLASSIFIERS:

            subset = details_result[
                details_result[
                    "classifier"
                ] == classifier
            ]

            if len(subset) == 0:
                continue

            summary_rows.append({
                "classifier": classifier,
                "test_datasets": subset[
                    "dataset"
                ].nunique(),
                "mean_selected_improvement": subset[
                    "selected_improvement"
                ].mean(),
                "mean_oracle_improvement": subset[
                    "oracle_improvement"
                ].mean(),
                "mean_gap_to_oracle": subset[
                    "regret"
                ].mean(),
                "mean_regret": subset[
                    "regret"
                ].mean(),
                "top1_accuracy": subset[
                    "top1_accuracy"
                ].mean(),
                "top3_accuracy": subset[
                    "top3_accuracy"
                ].mean(),
                "mean_spearman": subset[
                    "spearman"
                ].mean(),
                "mean_kendall": subset[
                    "kendall"
                ].mean()
            })

    summary_result = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_DIR /
        "nested_test_summary.csv"
    )

    summary_result.to_csv(
        summary_path,
        index=False
    )

    print("\n" + "=" * 75)
    print(
        "NESTED META-LEARNING SUMMARY"
    )
    print("=" * 75)

    if len(summary_result) > 0:
        print(
            summary_result.to_string(
                index=False
            )
        )
    else:
        print(
            "No valid test results generated."
        )

    print("\nSaved files:")

    print(
        split_path.resolve()
    )

    print(
        validation_path.resolve()
    )

    print(
        selected_path.resolve()
    )

    print(
        predictions_path.resolve()
    )

    print(
        details_path.resolve()
    )

    print(
        summary_path.resolve()
    )


if __name__ == "__main__":
    main()