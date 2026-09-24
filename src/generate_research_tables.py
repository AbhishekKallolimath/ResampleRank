from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "paper_tables"

META_DATASET_PATH = RESULTS_DIR / "meta_dataset.csv"
META_MODEL_SUMMARY_PATH = (
    RESULTS_DIR / "meta_model" / "meta_model_summary.csv"
)
SELECTION_SUMMARY_PATH = (
    RESULTS_DIR
    / "selection_performance"
    / "selection_performance_summary.csv"
)
SAVINGS_SUMMARY_PATH = (
    RESULTS_DIR
    / "computational_savings"
    / "computational_savings_summary.csv"
)
FRIEDMAN_PATH = (
    RESULTS_DIR
    / "statistical_analysis"
    / "friedman_results.csv"
)
POSTHOC_PATH = (
    RESULTS_DIR
    / "statistical_analysis"
    / "posthoc_results.csv"
)

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

STRATEGIES = [
    "No Resampling",
    "Random Oversampling",
    "Random Undersampling",
    "SMOTE",
    "ADASYN",
    "SMOTE-Tomek",
    "SMOTE-ENN",
]


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return pd.read_csv(path)


def table_01_dataset_characteristics():
    df = load_csv(META_DATASET_PATH)

    columns = [
        "dataset",
        "samples",
        "features",
        "numeric_features",
        "categorical_features",
        "missing_values",
        "duplicates",
        "majority_count",
        "minority_count",
        "minority_proportion",
        "imbalance_ratio",
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    result = (
        df[available]
        .drop_duplicates("dataset")
        .sort_values(
            "dataset"
        )
        .reset_index(drop=True)
    )

    return result


def table_02_resampling_performance():
    df = load_csv(META_DATASET_PATH)

    grouped = (
        df.groupby(
            ["resampling", "classifier"],
            as_index=False
        )["target_macro_f1"]
        .mean()
    )

    result = grouped.pivot(
        index="resampling",
        columns="classifier",
        values="target_macro_f1"
    )

    for classifier in CLASSIFIERS:
        if classifier not in result.columns:
            result[classifier] = pd.NA

    result = result[
        CLASSIFIERS
    ]

    result["Mean"] = result.mean(
        axis=1
    )

    result = result.reindex(
        STRATEGIES
    )

    result = result.reset_index()

    return result


def table_03_meta_model_performance():
    df = load_csv(
        META_MODEL_SUMMARY_PATH
    )

    columns = [
        "model",
        "classifier",
        "rows",
        "datasets",
        "mae",
        "rmse",
        "top1_resampling_accuracy",
        "top3_resampling_accuracy",
        "mean_spearman",
        "mean_kendall",
        "mean_regret",
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    result = df[available].copy()

    if "top1_resampling_accuracy" in result.columns:
        result[
            "top1_resampling_accuracy_percent"
        ] = (
            result[
                "top1_resampling_accuracy"
            ] * 100
        )

        result = result.drop(
            columns=[
                "top1_resampling_accuracy"
            ]
        )

    if "top3_resampling_accuracy" in result.columns:
        result[
            "top3_resampling_accuracy_percent"
        ] = (
            result[
                "top3_resampling_accuracy"
            ] * 100
        )

        result = result.drop(
            columns=[
                "top3_resampling_accuracy"
            ]
        )

    return result


def table_04_selection_performance():
    df = load_csv(
        SELECTION_SUMMARY_PATH
    )

    columns = [
        "meta_model",
        "classifier",
        "datasets",
        "mean_baseline_macro_f1",
        "mean_selected_macro_f1",
        "mean_oracle_macro_f1",
        "mean_selected_improvement",
        "mean_oracle_improvement",
        "mean_gap_to_oracle",
        "mean_regret",
        "top1_accuracy",
        "top3_accuracy",
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    result = df[available].copy()

    if "top1_accuracy" in result.columns:
        result[
            "top1_accuracy_percent"
        ] = result["top1_accuracy"]

        result = result.drop(
            columns=["top1_accuracy"]
        )

    if "top3_accuracy" in result.columns:
        result[
            "top3_accuracy_percent"
        ] = result["top3_accuracy"]

        result = result.drop(
            columns=["top3_accuracy"]
        )

    return result


def table_05_computational_savings():
    df = load_csv(
        SAVINGS_SUMMARY_PATH
    )

    columns = [
        "meta_model",
        "classifier",
        "datasets",
        "mean_full_evaluation_time",
        "mean_selected_strategy_time",
        "mean_savings_seconds",
        "mean_savings_percent",
        "mean_regret",
        "top1_accuracy",
        "top3_accuracy",
    ]

    available = [
        column
        for column in columns
        if column in df.columns
    ]

    result = df[available].copy()

    return result


def table_06_statistical_significance():
    df = load_csv(
        FRIEDMAN_PATH
    )

    result = df.copy()

    result[
        "significant_alpha_0_05"
    ] = (
        result["p_value"] < 0.05
    )

    return result


def table_07_posthoc_comparisons():
    df = load_csv(
        POSTHOC_PATH
    )

    if "holm_adjusted_p_value" not in df.columns:
        raise ValueError(
            "holm_adjusted_p_value "
            "column is missing from posthoc results."
        )

    significant = df[
        df[
            "holm_adjusted_p_value"
        ] < 0.05
    ].copy()

    columns = [
        "analysis",
        "strategy_a",
        "strategy_b",
        "datasets",
        "mean_a",
        "mean_b",
        "mean_difference_a_minus_b",
        "raw_p_value",
        "holm_adjusted_p_value",
        "rank_biserial_effect",
    ]

    available = [
        column
        for column in columns
        if column in significant.columns
    ]

    return significant[available]


def save_table(df, filename):
    path = OUTPUT_DIR / filename

    df.to_csv(
        path,
        index=False
    )

    print(f"Saved: {path}")


def main():
    print("=" * 75)
    print(
        "ResampleRank - Research Paper Table Generation"
    )
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nGenerating paper-ready tables...\n")

    table_01 = table_01_dataset_characteristics()
    save_table(
        table_01,
        "table_01_dataset_characteristics.csv"
    )

    table_02 = table_02_resampling_performance()
    save_table(
        table_02,
        "table_02_resampling_performance.csv"
    )

    table_03 = table_03_meta_model_performance()
    save_table(
        table_03,
        "table_03_meta_model_performance.csv"
    )

    table_04 = table_04_selection_performance()
    save_table(
        table_04,
        "table_04_selection_performance.csv"
    )

    table_05 = table_05_computational_savings()
    save_table(
        table_05,
        "table_05_computational_savings.csv"
    )

    table_06 = table_06_statistical_significance()
    save_table(
        table_06,
        "table_06_statistical_significance.csv"
    )

    table_07 = table_07_posthoc_comparisons()
    save_table(
        table_07,
        "table_07_posthoc_comparisons.csv"
    )

    print("\n" + "=" * 75)
    print(
        "TABLE GENERATION COMPLETE"
    )
    print("=" * 75)

    print(
        f"\nGenerated 7 tables in: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()