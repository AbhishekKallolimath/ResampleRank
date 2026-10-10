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
    GradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parents[1]

META_DATASET_FILE = (
    BASE_DIR / "results" / "meta_dataset_clean43.csv"
)

OUTPUT_DIR = BASE_DIR / "results" / "meta_model_clean43"

RANDOM_STATE = 42

EXPECTED_SPLIT_COUNTS = {
    "TRAIN": 23,
    "VALIDATION": 14,
    "TEST": 6,
}

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
    "class_entropy",
    "mean_feature_variance",
    "mean_feature_std",
    "mean_abs_feature_skewness",
    "mean_unique_value_ratio",
    "mean_abs_correlation",
    "max_abs_correlation",
    "mean_zero_proportion",
    "mean_mutual_information",
    "max_mutual_information",
    "one_nn_error_rate",
    "minority_majority_neighbor_ratio",
    "minority_mean_neighbor_distance",
]

CATEGORICAL_FEATURES = ["resampling_code"]

TARGET = "delta_macro_f1_vs_baseline"

CLASSIFIERS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
]


def build_models():
    return {
        "ridge": Ridge(alpha=1.0),

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

        "xgboost": XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            objective="reg:squarederror",
            verbosity=0,
        ),
    }


def build_pipeline(model):
    preprocessor = ColumnTransformer(
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

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def load_meta_dataset():
    if not META_DATASET_FILE.exists():
        raise FileNotFoundError(
            "Meta-dataset not found: {}".format(
                META_DATASET_FILE
            )
        )

    df = pd.read_csv(META_DATASET_FILE)

    required = [
        "dataset",
        "split",
        "classifier",
        "resampling",
        "resampling_code",
        "target_macro_f1",
        "baseline_macro_f1",
        "delta_macro_f1_vs_baseline",
        "verification",
    ] + NUMERIC_FEATURES

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: {}".format(missing)
        )

    if len(df) != 1204:
        raise ValueError(
            "Expected 1204 rows, found {}".format(len(df))
        )

    if df["dataset"].nunique() != 43:
        raise ValueError(
            "Expected 43 unique datasets, found {}".format(
                df["dataset"].nunique()
            )
        )

    df["split"] = (
        df["split"].astype(str).str.strip().str.upper()
    )

    split_manifest = (
        df[["dataset", "split"]].drop_duplicates()
    )

    if split_manifest["dataset"].duplicated().any():
        raise ValueError(
            "At least one dataset has multiple split assignments."
        )

    split_counts = (
        split_manifest["split"].value_counts().to_dict()
    )

    if split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            "Unexpected split counts. Expected {}; found {}.".format(
                EXPECTED_SPLIT_COUNTS, split_counts
            )
        )

    if not df["verification"].astype(str).str.upper().eq("PASS").all():
        raise ValueError(
            "Only fully verified experiments may be used."
        )

    if df.duplicated(
        ["dataset", "resampling_code", "classifier"]
    ).any():
        raise ValueError(
            "Duplicate dataset/classifier/resampling combinations found."
        )

    if not df.groupby("dataset").size().eq(28).all():
        raise ValueError(
            "Every dataset must contain exactly 28 experiments."
        )

    required_values = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + [
            TARGET,
            "target_macro_f1",
            "baseline_macro_f1",
        ]
    )

    if df[required_values].isna().any().any():
        missing_counts = (
            df[required_values].isna().sum()
        )
        raise ValueError(
            "Missing values detected:\n{}".format(
                missing_counts[missing_counts > 0].to_string()
            )
        )

    numeric_values = df[NUMERIC_FEATURES].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Numeric meta-features contain non-finite values."
        )

    expected_classifiers = set(CLASSIFIERS)

    if set(df["classifier"].unique()) != expected_classifiers:
        raise ValueError(
            "Classifier set differs from the expected four classifiers."
        )

    return df, split_manifest


def safe_correlations(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if (
        len(actual) < 2
        or len(predicted) < 2
        or len(np.unique(actual)) < 2
        or len(np.unique(predicted)) < 2
    ):
        return np.nan, np.nan

    sp = spearmanr(actual, predicted)[0]
    kt = kendalltau(actual, predicted)[0]

    return (
        float(sp) if np.isfinite(sp) else np.nan,
        float(kt) if np.isfinite(kt) else np.nan,
    )


def evaluate_dataset_classifier(group):
    group = group.copy()

    if len(group) != 7 or group["resampling_code"].nunique() != 7:
        raise ValueError(
            "{} / {} does not contain seven unique strategies.".format(
                group.iloc[0]["dataset"],
                group.iloc[0]["classifier"],
            )
        )

    actual_scores = group["target_macro_f1"].to_numpy(dtype=float)
    predicted_scores = group["predicted_macro_f1"].to_numpy(dtype=float)

    actual_max = float(np.max(actual_scores))

    best_mask = np.isclose(
        actual_scores,
        actual_max,
        rtol=0.0,
        atol=1e-12,
    )

    actual_best_codes = group.loc[
        best_mask, "resampling_code"
    ].tolist()

    ranked = group.sort_values(
        ["predicted_macro_f1", "resampling_code"],
        ascending=[False, True],
        kind="mergesort",
    )

    predicted_best_code = ranked.iloc[0]["resampling_code"]

    top3_codes = set(
        ranked.head(3)["resampling_code"].tolist()
    )

    selected_row = ranked.iloc[0]

    baseline_values = group["baseline_macro_f1"].to_numpy(
        dtype=float
    )

    if not np.allclose(
        baseline_values,
        baseline_values[0],
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Baseline Macro-F1 is inconsistent within a dataset/classifier group."
        )

    baseline = float(baseline_values[0])
    selected_score = float(selected_row["target_macro_f1"])

    sp, kt = safe_correlations(
        actual_scores,
        predicted_scores,
    )

    return {
        "dataset": group.iloc[0]["dataset"],
        "split": group.iloc[0]["split"],
        "classifier": group.iloc[0]["classifier"],
        "number_of_resampling_strategies": len(group),
        "actual_best_resampling_codes": "|".join(
            actual_best_codes
        ),
        "predicted_best_resampling_code": predicted_best_code,
        "actual_best_macro_f1": actual_max,
        "selected_actual_macro_f1": selected_score,
        "baseline_macro_f1": baseline,
        "selected_improvement": selected_score - baseline,
        "oracle_improvement": actual_max - baseline,
        "regret": actual_max - selected_score,
        "top1_correct": predicted_best_code in actual_best_codes,
        "top3_correct": bool(top3_codes.intersection(actual_best_codes)),
        "spearman": sp,
        "kendall": kt,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("ResampleRank - Fixed Clean-43 Meta-Learning")
    print("=" * 75)

    df, split_manifest = load_meta_dataset()

    print("\nMeta-dataset:", META_DATASET_FILE)
    print("Rows:", len(df))
    print("Datasets:", df["dataset"].nunique())
    print("\nSplit counts:")
    print(split_manifest["split"].value_counts().reindex(
        ["TRAIN", "VALIDATION", "TEST"],
        fill_value=0,
    ).to_string())

    manifest_path = (
        OUTPUT_DIR / "dataset_split_manifest.csv"
    )
    split_manifest.sort_values(
        ["split", "dataset"]
    ).to_csv(manifest_path, index=False)

    validation_results = []
    selected_model_results = []
    test_prediction_frames = []
    test_selection_results = []

    for classifier in CLASSIFIERS:
        print("\n" + "-" * 75)
        print("Classifier:", classifier)
        print("-" * 75)

        classifier_df = df[
            df["classifier"] == classifier
        ].copy()

        train_df = classifier_df[
            classifier_df["split"] == "TRAIN"
        ].copy()

        validation_df = classifier_df[
            classifier_df["split"] == "VALIDATION"
        ].copy()

        test_df = classifier_df[
            classifier_df["split"] == "TEST"
        ].copy()

        if (
            train_df["dataset"].nunique() != 23
            or validation_df["dataset"].nunique() != 14
            or test_df["dataset"].nunique() != 6
        ):
            raise ValueError(
                "Incorrect split counts for classifier {}.".format(
                    classifier
                )
            )

        X_train = train_df[
            NUMERIC_FEATURES + CATEGORICAL_FEATURES
        ]
        y_train = train_df[TARGET]

        X_validation = validation_df[
            NUMERIC_FEATURES + CATEGORICAL_FEATURES
        ]
        y_validation = validation_df[TARGET]

        candidate_results = []

        for model_name, model in build_models().items():
            print("Validation candidate:", model_name)

            result = {
                "classifier": classifier,
                "meta_model": model_name,
                "train_datasets": train_df["dataset"].nunique(),
                "validation_datasets": validation_df["dataset"].nunique(),
                "test_datasets": test_df["dataset"].nunique(),
            }

            try:
                pipeline = build_pipeline(model)
                pipeline.fit(X_train, y_train)

                val_pred = pipeline.predict(X_validation)

                val_mae = float(
                    mean_absolute_error(y_validation, val_pred)
                )
                val_rmse = float(
                    np.sqrt(mean_squared_error(y_validation, val_pred))
                )

                result["validation_mae"] = val_mae
                result["validation_rmse"] = val_rmse
                result["error"] = ""

                candidate_results.append(result)

                print(
                    "  validation MAE={:.6f}, RMSE={:.6f}".format(
                        val_mae, val_rmse
                    )
                )

            except Exception as exc:
                result["validation_mae"] = np.nan
                result["validation_rmse"] = np.nan
                result["error"] = str(exc)
                candidate_results.append(result)

                print("  MODEL ERROR:", exc)

        classifier_validation = pd.DataFrame(candidate_results)
        validation_results.extend(candidate_results)

        valid_candidates = classifier_validation.dropna(
            subset=["validation_mae"]
        ).sort_values(
            ["validation_mae", "validation_rmse", "meta_model"],
            ascending=[True, True, True],
            kind="mergesort",
        )

        if valid_candidates.empty:
            raise RuntimeError(
                "All candidate models failed for classifier {}.".format(
                    classifier
                )
            )

        chosen = valid_candidates.iloc[0]
        selected_name = chosen["meta_model"]

        print(
            "\nSelected on validation: {} (MAE={:.6f})".format(
                selected_name,
                chosen["validation_mae"],
            )
        )

        selected_model_results.append({
            "classifier": classifier,
            "selected_meta_model": selected_name,
            "validation_mae": float(chosen["validation_mae"]),
            "validation_rmse": float(chosen["validation_rmse"]),
            "train_datasets": train_df["dataset"].nunique(),
            "validation_datasets": validation_df["dataset"].nunique(),
            "test_datasets": test_df["dataset"].nunique(),
        })

        # After model selection, refit on TRAIN + VALIDATION.
        combined_train = pd.concat(
            [train_df, validation_df],
            ignore_index=True,
        )

        final_pipeline = build_pipeline(
            build_models()[selected_name]
        )

        final_pipeline.fit(
            combined_train[
                NUMERIC_FEATURES + CATEGORICAL_FEATURES
            ],
            combined_train[TARGET],
        )

        X_test = test_df[
            NUMERIC_FEATURES + CATEGORICAL_FEATURES
        ]

        predicted_delta = final_pipeline.predict(X_test)

        prediction_df = test_df.copy()
        prediction_df["predicted_delta_macro_f1"] = predicted_delta
        prediction_df["predicted_macro_f1"] = (
            prediction_df["baseline_macro_f1"]
            + prediction_df["predicted_delta_macro_f1"]
        ).clip(lower=0.0, upper=1.0)

        prediction_df["selected_meta_model"] = selected_name

        test_prediction_frames.append(prediction_df)

        y_test = test_df[TARGET].to_numpy(dtype=float)

        test_mae = float(
            mean_absolute_error(y_test, predicted_delta)
        )
        test_rmse = float(
            np.sqrt(mean_squared_error(y_test, predicted_delta))
        )

        print(
            "Held-out test MAE={:.6f}, RMSE={:.6f}".format(
                test_mae, test_rmse
            )
        )

        for (_, _), group in prediction_df.groupby(
            ["dataset", "classifier"],
            sort=True,
        ):
            test_selection_results.append(
                evaluate_dataset_classifier(group)
            )

    validation_df_out = pd.DataFrame(validation_results)
    selected_df_out = pd.DataFrame(selected_model_results)
    predictions_df_out = pd.concat(
        test_prediction_frames,
        ignore_index=True,
    )
    selection_df_out = pd.DataFrame(test_selection_results)

    # Create test summaries only after final predictions are complete.
    summary_rows = []

    for classifier in CLASSIFIERS:
        pred_group = predictions_df_out[
            predictions_df_out["classifier"] == classifier
        ]
        selection_group = selection_df_out[
            selection_df_out["classifier"] == classifier
        ]

        summary_rows.append({
            "classifier": classifier,
            "selected_meta_model": selected_df_out.loc[
                selected_df_out["classifier"] == classifier,
                "selected_meta_model",
            ].iloc[0],
            "test_datasets": selection_group["dataset"].nunique(),
            "test_experiment_rows": len(pred_group),
            "mae_delta": mean_absolute_error(
                pred_group[TARGET],
                pred_group["predicted_delta_macro_f1"],
            ),
            "rmse_delta": np.sqrt(mean_squared_error(
                pred_group[TARGET],
                pred_group["predicted_delta_macro_f1"],
            )),
            "top1_accuracy": selection_group["top1_correct"].mean(),
            "top3_accuracy": selection_group["top3_correct"].mean(),
            "mean_spearman": selection_group["spearman"].mean(),
            "mean_kendall": selection_group["kendall"].mean(),
            "mean_regret": selection_group["regret"].mean(),
            "mean_selected_improvement": selection_group[
                "selected_improvement"
            ].mean(),
            "mean_oracle_improvement": selection_group[
                "oracle_improvement"
            ].mean(),
        })

    summary_rows.append({
        "classifier": "ALL",
        "selected_meta_model": "Per-classifier validation selection",
        "test_datasets": selection_df_out["dataset"].nunique(),
        "test_experiment_rows": len(predictions_df_out),
        "mae_delta": mean_absolute_error(
            predictions_df_out[TARGET],
            predictions_df_out["predicted_delta_macro_f1"],
        ),
        "rmse_delta": np.sqrt(mean_squared_error(
            predictions_df_out[TARGET],
            predictions_df_out["predicted_delta_macro_f1"],
        )),
        "top1_accuracy": selection_df_out["top1_correct"].mean(),
        "top3_accuracy": selection_df_out["top3_correct"].mean(),
        "mean_spearman": selection_df_out["spearman"].mean(),
        "mean_kendall": selection_df_out["kendall"].mean(),
        "mean_regret": selection_df_out["regret"].mean(),
        "mean_selected_improvement": selection_df_out[
            "selected_improvement"
        ].mean(),
        "mean_oracle_improvement": selection_df_out[
            "oracle_improvement"
        ].mean(),
    })

    summary_df = pd.DataFrame(summary_rows)

    # Validate final predictions before writing the result files.
    if len(predictions_df_out) != 168:
        raise RuntimeError(
            "Expected 168 held-out prediction rows; found {}.".format(
                len(predictions_df_out)
            )
        )

    if predictions_df_out["dataset"].nunique() != 6:
        raise RuntimeError(
            "Expected exactly six held-out test datasets."
        )

    if not predictions_df_out["split"].eq("TEST").all():
        raise RuntimeError(
            "Non-test datasets appeared in held-out predictions."
        )

    if len(selection_df_out) != 24:
        raise RuntimeError(
            "Expected 24 test dataset/classifier ranking evaluations; "
            "found {}.".format(len(selection_df_out))
        )

    if not np.isfinite(
        predictions_df_out["predicted_delta_macro_f1"].to_numpy()
    ).all():
        raise RuntimeError(
            "Final test predictions contain non-finite values."
        )

    validation_path = (
        OUTPUT_DIR / "validation_model_comparison.csv"
    )
    selected_path = (
        OUTPUT_DIR / "selected_models_by_classifier.csv"
    )
    prediction_path = (
        OUTPUT_DIR / "test_predictions.csv"
    )
    selection_path = (
        OUTPUT_DIR / "test_selection_details.csv"
    )
    summary_path = (
        OUTPUT_DIR / "test_summary.csv"
    )

    validation_df_out.to_csv(validation_path, index=False)
    selected_df_out.to_csv(selected_path, index=False)
    predictions_df_out.to_csv(prediction_path, index=False)
    selection_df_out.to_csv(selection_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 75)
    print("FIXED-SPLIT META-LEARNING SUMMARY")
    print("=" * 75)

    print("\nValidation model comparison:")
    print(validation_df_out.to_string(index=False))

    print("\nSelected models:")
    print(selected_df_out.to_string(index=False))

    print("\nHeld-out test summary:")
    print(summary_df.to_string(index=False))

    print("\nSaved files:")
    print(manifest_path)
    print(validation_path)
    print(selected_path)
    print(prediction_path)
    print(selection_path)
    print(summary_path)

    print("\nTRAIN/VALIDATION/TEST protocol completed.")
    print("Test datasets were not used for model selection.")


if __name__ == "__main__":
    main()
