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
    / "computational_savings"
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
            f"Runtime file not found: {file_path}"
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

        full_evaluation_time = (
            runtime_df[
                "execution_time_seconds"
            ]
            .sum()
        )

        selected_rows = runtime_df[
            runtime_df[
                "resampling_code"
            ]
            == predicted_strategy
        ]

        if selected_rows.empty:
            continue

        selected_strategy_time = float(
            selected_rows.iloc[0][
                "execution_time_seconds"
            ]
        )

        savings_seconds = (
            full_evaluation_time
            - selected_strategy_time
        )

        savings_percent = (
            savings_seconds
            / full_evaluation_time
            * 100.0
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
                "regret": ranking_row[
                    "regret"
                ],
                "top1_correct": ranking_row[
                    "top1_correct"
                ],
                "top3_correct": ranking_row[
                    "top3_correct"
                ],
                "full_evaluation_time_seconds": (
                    full_evaluation_time
                ),
                "selected_strategy_time_seconds": (
                    selected_strategy_time
                ),
                "savings_seconds": (
                    savings_seconds
                ),
                "savings_percent": (
                    savings_percent
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
            "No computational savings results were generated."
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
            mean_full_evaluation_time=(
                "full_evaluation_time_seconds",
                "mean",
            ),
            mean_selected_strategy_time=(
                "selected_strategy_time_seconds",
                "mean",
            ),
            mean_savings_seconds=(
                "savings_seconds",
                "mean",
            ),
            mean_savings_percent=(
                "savings_percent",
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
        / "computational_savings_details.csv"
    )

    summary_file = (
        OUTPUT_DIR
        / "computational_savings_summary.csv"
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
        "ResampleRank - Computational Savings Analysis"
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