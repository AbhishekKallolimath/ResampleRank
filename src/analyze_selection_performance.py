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
    / "selection_performance"
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


def load_ranking_file(
    meta_model,
    classifier,
):
    file_path = (
        META_MODEL_DIR
        / f"{meta_model}_{classifier}_ranking.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Ranking file not found: {file_path}"
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
    ranking_df = load_ranking_file(
        meta_model,
        classifier,
    )

    results = []

    for _, ranking_row in ranking_df.iterrows():

        dataset = ranking_row[
            "dataset"
        ]

        predicted_strategy = ranking_row[
            "predicted_best_resampling"
        ]

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

        baseline_rows = runtime_df[
            runtime_df[
                "resampling_code"
            ] == "none"
        ]

        selected_rows = runtime_df[
            runtime_df[
                "resampling_code"
            ] == predicted_strategy
        ]

        if (
            baseline_rows.empty
            or selected_rows.empty
        ):
            continue

        baseline_macro_f1 = float(
            baseline_rows.iloc[0][
                "macro_f1_math"
            ]
        )

        selected_macro_f1 = float(
            selected_rows.iloc[0][
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

        oracle_gap = (
            oracle_macro_f1
            - selected_macro_f1
        )

        results.append(
            {
                "dataset": dataset,
                "classifier": classifier,
                "meta_model": meta_model,
                "predicted_resampling": predicted_strategy,
                "actual_best_resampling": ranking_row[
                    "actual_best_resampling"
                ],
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
                    oracle_gap
                ),
                "regret": ranking_row[
                    "regret"
                ],
                "top1_correct": ranking_row[
                    "top1_correct"
                ],
                "top3_correct": ranking_row[
                    "top3_correct"
                ],
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
            "No selection-performance results were generated."
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
            mean_regret=(
                "regret",
                "mean",
            ),
            top1_accuracy=(
                "top1_correct",
                "mean",
            ),
            top3_accuracy=(
                "top3_correct",
                "mean",
            ),
        )
        .reset_index()
    )

    summary_df[
        "top1_accuracy"
    ] *= 100.0

    summary_df[
        "top3_accuracy"
    ] *= 100.0

    details_file = (
        OUTPUT_DIR
        / "selection_performance_details.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / "selection_performance_summary.csv"
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
        "ResampleRank - Selection Performance Analysis"
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