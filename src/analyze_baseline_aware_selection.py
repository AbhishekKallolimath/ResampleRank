from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    BASE_DIR
    / "results"
)

META_MODEL_DIR = (
    RESULTS_DIR
    / "meta_model"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "baseline_aware_selection"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


META_MODELS = [
    "ridge",
    "random_forest",
    "gradient_boosting",
    "xgboost",
]

CLASSIFIERS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
]


def load_prediction_file(
    meta_model,
    classifier,
):
    file_path = (
        META_MODEL_DIR
        / f"{meta_model}_{classifier}_predictions.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {file_path}"
        )

    return pd.read_csv(
        file_path
    )


def load_runtime_file(
    dataset,
    classifier,
):
    file_path = (
        RESULTS_DIR
        / f"{dataset}_{classifier}_results.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Result file not found: {file_path}"
        )

    return pd.read_csv(
        file_path
    )


def analyze_model_classifier(
    meta_model,
    classifier,
):
    prediction_df = load_prediction_file(
        meta_model,
        classifier,
    )

    prediction_df[
        "predicted_delta"
    ] = (
        prediction_df[
            "predicted_macro_f1"
        ]
        - prediction_df[
            "baseline_macro_f1"
        ]
    )

    results = []

    for dataset, group in prediction_df.groupby(
        "dataset"
    ):

        positive_predictions = group[
            group["predicted_delta"] > 0
        ].copy()

        if positive_predictions.empty:
            selected_strategy = "none"
            selected_prediction = (
                group[
                    group[
                        "resampling_code"
                    ] == "none"
                ]
                .iloc[0]
            )
        else:
            selected_prediction = (
                positive_predictions
                .sort_values(
                    by=[
                        "predicted_delta",
                        "resampling_code",
                    ],
                    ascending=[
                        False,
                        True,
                    ],
                )
                .iloc[0]
            )

            selected_strategy = (
                selected_prediction[
                    "resampling_code"
                ]
            )

        runtime_df = load_runtime_file(
            dataset,
            classifier,
        )

        runtime_df = runtime_df[
            runtime_df[
                "verification"
            ] == "PASS"
        ].copy()

        if runtime_df.empty:
            continue

        baseline_row = runtime_df[
            runtime_df[
                "resampling_code"
            ] == "none"
        ]

        selected_row = runtime_df[
            runtime_df[
                "resampling_code"
            ] == selected_strategy
        ]

        if (
            baseline_row.empty
            or selected_row.empty
        ):
            continue

        baseline_macro_f1 = float(
            baseline_row.iloc[0][
                "macro_f1_math"
            ]
        )

        selected_macro_f1 = float(
            selected_row.iloc[0][
                "macro_f1_math"
            ]
        )

        oracle_macro_f1 = float(
            runtime_df[
                "macro_f1_math"
            ].max()
        )

        selected_improvement = (
            selected_macro_f1
            - baseline_macro_f1
        )

        oracle_improvement = (
            oracle_macro_f1
            - baseline_macro_f1
        )

        gap_to_oracle = (
            oracle_macro_f1
            - selected_macro_f1
        )

        selected_runtime = float(
            selected_row.iloc[0][
                "execution_time_seconds"
            ]
        )

        full_evaluation_runtime = float(
            runtime_df[
                "execution_time_seconds"
            ].sum()
        )

        savings_seconds = (
            full_evaluation_runtime
            - selected_runtime
        )

        savings_percent = (
            savings_seconds
            / full_evaluation_runtime
            * 100.0
        )

        results.append(
            {
                "dataset": dataset,
                "classifier": classifier,
                "meta_model": meta_model,
                "predicted_selected_resampling": (
                    selected_strategy
                ),
                "predicted_delta_for_selection": (
                    float(
                        selected_prediction[
                            "predicted_delta"
                        ]
                    )
                ),
                "baseline_macro_f1": (
                    baseline_macro_f1
                ),
                "selected_macro_f1": (
                    selected_macro_f1
                ),
                "oracle_best_macro_f1": (
                    oracle_macro_f1
                ),
                "selected_improvement_vs_baseline": (
                    selected_improvement
                ),
                "oracle_improvement_vs_baseline": (
                    oracle_improvement
                ),
                "gap_to_oracle": (
                    gap_to_oracle
                ),
                "selected_runtime_seconds": (
                    selected_runtime
                ),
                "full_evaluation_runtime_seconds": (
                    full_evaluation_runtime
                ),
                "savings_seconds": (
                    savings_seconds
                ),
                "savings_percent": (
                    savings_percent
                ),
                "positive_selection": (
                    selected_strategy != "none"
                ),
                "improves_over_baseline": (
                    selected_improvement > 0
                ),
                "matches_oracle": (
                    selected_macro_f1
                    == oracle_macro_f1
                ),
            }
        )

    return pd.DataFrame(
        results
    )


def main():

    all_results = []

    for meta_model in META_MODELS:

        for classifier in CLASSIFIERS:

            result_df = (
                analyze_model_classifier(
                    meta_model,
                    classifier,
                )
            )

            all_results.append(
                result_df
            )

    if not all_results:
        raise ValueError(
            "No baseline-aware selection results were generated."
        )

    details_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    summary_df = (
        details_df
        .groupby(
            [
                "meta_model",
                "classifier",
            ]
        )
        .agg(
            datasets=(
                "dataset",
                "nunique",
            ),
            positive_selection_rate=(
                "positive_selection",
                "mean",
            ),
            improvement_rate=(
                "improves_over_baseline",
                "mean",
            ),
            oracle_match_rate=(
                "matches_oracle",
                "mean",
            ),
            mean_baseline_macro_f1=(
                "baseline_macro_f1",
                "mean",
            ),
            mean_selected_macro_f1=(
                "selected_macro_f1",
                "mean",
            ),
            mean_oracle_macro_f1=(
                "oracle_best_macro_f1",
                "mean",
            ),
            mean_selected_improvement=(
                "selected_improvement_vs_baseline",
                "mean",
            ),
            mean_oracle_improvement=(
                "oracle_improvement_vs_baseline",
                "mean",
            ),
            mean_gap_to_oracle=(
                "gap_to_oracle",
                "mean",
            ),
            mean_savings_percent=(
                "savings_percent",
                "mean",
            ),
        )
        .reset_index()
    )

    summary_df[
        "positive_selection_rate"
    ] *= 100.0

    summary_df[
        "improvement_rate"
    ] *= 100.0

    summary_df[
        "oracle_match_rate"
    ] *= 100.0

    details_file = (
        OUTPUT_DIR
        / "baseline_aware_selection_details.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / "baseline_aware_selection_summary.csv"
    )

    details_df.to_csv(
        details_file,
        index=False,
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    print(
        "=" * 75
    )

    print(
        "ResampleRank - Baseline-Aware Selection Analysis"
    )

    print(
        "=" * 75
    )

    print(
        f"Detailed rows : {len(details_df)}"
    )

    print(
        f"Meta-models   : {details_df['meta_model'].nunique()}"
    )

    print(
        f"Classifiers   : {details_df['classifier'].nunique()}"
    )

    print(
        "\nSUMMARY"
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
        details_file
    )

    print(
        summary_file
    )


if __name__ == "__main__":
    main()