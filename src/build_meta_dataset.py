from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results"
PROFILE_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "dataset_profiles.csv"
)

OUTPUT_FILE = RESULTS_DIR / "meta_dataset.csv"


DATASET_RESULT_FILES = {
    "pima": [
        RESULTS_DIR / "pima_resampling_results.csv",
        RESULTS_DIR / "pima_decision_tree_results.csv",
        RESULTS_DIR / "pima_random_forest_results.csv",
        RESULTS_DIR / "pima_xgboost_results.csv",
    ],
    "glass1": [
        RESULTS_DIR / "glass1_logistic_regression_results.csv",
        RESULTS_DIR / "glass1_decision_tree_results.csv",
        RESULTS_DIR / "glass1_random_forest_results.csv",
        RESULTS_DIR / "glass1_xgboost_results.csv",
    ],
}


def load_result_files():
    frames = []

    for dataset, result_files in DATASET_RESULT_FILES.items():
        print("\n" + "-" * 75)
        print(f"Loading results for dataset: {dataset}")
        print("-" * 75)

        for file_path in result_files:

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Required result file not found: {file_path}"
                )

            df = pd.read_csv(file_path)

            if df.empty:
                raise ValueError(
                    f"Result file is empty: {file_path}"
                )

            print(
                f"{file_path.name} -> "
                f"{len(df)} rows"
            )

            frames.append(df)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def validate_experiments(meta_df):

    expected_rows_per_dataset = 28

    for dataset in DATASET_RESULT_FILES:

        dataset_df = meta_df[
            meta_df["dataset"] == dataset
        ]

        if len(dataset_df) != expected_rows_per_dataset:
            raise ValueError(
                f"{dataset}: expected "
                f"{expected_rows_per_dataset} experiments, "
                f"found {len(dataset_df)}"
            )

        duplicates = dataset_df.duplicated(
            subset=[
                "dataset",
                "resampling_code",
                "classifier",
            ]
        ).sum()

        if duplicates != 0:
            raise ValueError(
                f"{dataset}: found "
                f"{duplicates} duplicate experiments."
            )

    total_expected = (
        expected_rows_per_dataset
        * len(DATASET_RESULT_FILES)
    )

    if len(meta_df) != total_expected:
        raise ValueError(
            f"Expected {total_expected} total experiments, "
            f"found {len(meta_df)}"
        )

    verification_failures = (
        meta_df["verification"] != "PASS"
    ).sum()

    if verification_failures != 0:
        raise ValueError(
            f"Found {verification_failures} "
            f"mathematical verification failures."
        )


def add_dataset_meta_features(meta_df):

    if not PROFILE_FILE.exists():
        raise FileNotFoundError(
            f"Dataset profile not found: {PROFILE_FILE}"
        )

    profiles = pd.read_csv(
        PROFILE_FILE
    )

    if "dataset" not in profiles.columns:
        raise ValueError(
            "Dataset profiles must contain "
            "'dataset' column."
        )

    profiles = profiles.drop_duplicates(
        subset=["dataset"]
    )

    meta_df = meta_df.merge(
        profiles,
        on="dataset",
        how="left",
        suffixes=("", "_profile"),
    )

    required_profile_columns = [
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

    for column in required_profile_columns:

        if column not in meta_df.columns:
            raise ValueError(
                f"Missing meta-feature column: {column}"
            )

    for dataset in meta_df["dataset"].unique():

        rows = meta_df[
            meta_df["dataset"] == dataset
        ]

        if rows["imbalance_ratio"].isna().any():
            raise ValueError(
                f"Missing profile information "
                f"for dataset: {dataset}"
            )

    return meta_df


def add_meta_targets(meta_df):

    meta_df["target_macro_f1"] = (
        meta_df["macro_f1_math"]
    )

    baseline = meta_df[
        meta_df["resampling_code"] == "none"
    ][
        [
            "dataset",
            "classifier",
            "target_macro_f1",
        ]
    ].rename(
        columns={
            "target_macro_f1":
            "baseline_macro_f1"
        }
    )

    expected_baselines = (
        meta_df["dataset"].nunique()
        * meta_df["classifier"].nunique()
    )

    if len(baseline) != expected_baselines:
        raise ValueError(
            "Unexpected number of baseline rows."
        )

    meta_df = meta_df.merge(
        baseline,
        on=[
            "dataset",
            "classifier",
        ],
        how="left",
    )

    meta_df["delta_macro_f1_vs_baseline"] = (
        meta_df["target_macro_f1"]
        - meta_df["baseline_macro_f1"]
    )

    if meta_df[
        "baseline_macro_f1"
    ].isna().any():

        raise ValueError(
            "Some experiments are missing "
            "their no-resampling baseline."
        )

    return meta_df


def select_columns(meta_df):

    final_columns = [

        "dataset",

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

        "resampling_code",
        "resampling",
        "classifier",

        "tn",
        "fp",
        "fn",
        "tp",

        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",

        "f1_class_0_math",
        "f1_class_1_math",

        "macro_f1_code",
        "macro_f1_math",

        "metric_error",
        "verification",

        "execution_time_seconds",

        "target_macro_f1",

        "baseline_macro_f1",
        "delta_macro_f1_vs_baseline",
    ]

    missing_columns = [
        column
        for column in final_columns
        if column not in meta_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: "
            f"{missing_columns}"
        )

    return meta_df[final_columns]


def main():

    print("=" * 75)
    print(
        "ResampleRank - Building Combined Meta-Dataset"
    )
    print("=" * 75)

    meta_df = load_result_files()

    validate_experiments(
        meta_df
    )

    meta_df = add_dataset_meta_features(
        meta_df
    )

    meta_df = add_meta_targets(
        meta_df
    )

    meta_df = select_columns(
        meta_df
    )

    meta_df = meta_df.sort_values(
        by=[
            "dataset",
            "classifier",
            "macro_f1_math",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    meta_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 75)
    print("META-DATASET SUMMARY")
    print("=" * 75)

    print(
        f"Total experiments : "
        f"{len(meta_df)}"
    )

    print(
        f"Datasets          : "
        f"{meta_df['dataset'].nunique()}"
    )

    print(
        f"Resampling methods: "
        f"{meta_df['resampling'].nunique()}"
    )

    print(
        f"Classifiers       : "
        f"{meta_df['classifier'].nunique()}"
    )

    print(
        "\nExperiments per dataset:"
    )

    print(
        meta_df[
            "dataset"
        ].value_counts().sort_index()
    )

    verification_failures = (
        meta_df["verification"] != "PASS"
    ).sum()

    print(
        "\nMathematical verification failures: "
        f"{verification_failures}"
    )

    print("\nMacro-F1 range:")

    print(
        f"Minimum: "
        f"{meta_df['target_macro_f1'].min():.6f}"
    )

    print(
        f"Maximum: "
        f"{meta_df['target_macro_f1'].max():.6f}"
    )

    print("\nMeta-dataset preview:")

    print(
        meta_df[
            [
                "dataset",
                "resampling",
                "classifier",
                "imbalance_ratio",
                "target_macro_f1",
            ]
        ].head(20).to_string(
            index=False
        )
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()