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


PRIMARY_DATASETS = [
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


EXPECTED_EXPERIMENTS_PER_DATASET = 28


def get_result_files():
    result_files = {}

    for dataset in PRIMARY_DATASETS:
        result_files[dataset] = []

        for classifier in CLASSIFIERS:
            result_files[dataset].append(
                RESULTS_DIR
                / f"{dataset}_{classifier}_results.csv"
            )

    return result_files


DATASET_RESULT_FILES = get_result_files()


def load_result_files():
    frames = []

    print("\n" + "-" * 75)
    print("LOADING BENCHMARK RESULTS")
    print("-" * 75)

    for dataset, result_files in DATASET_RESULT_FILES.items():

        print(
            f"\nDataset: {dataset}"
        )

        for file_path in result_files:

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Required result file not found: "
                    f"{file_path}"
                )

            df = pd.read_csv(
                file_path
            )

            if df.empty:
                raise ValueError(
                    f"Result file is empty: "
                    f"{file_path}"
                )

            required_columns = [
                "dataset",
                "resampling_code",
                "resampling",
                "classifier",
                "macro_f1_math",
                "verification",
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in df.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"{file_path.name} is missing columns: "
                    f"{missing_columns}"
                )

            print(
                f"  {file_path.name} -> "
                f"{len(df)} rows"
            )

            frames.append(
                df
            )

    if not frames:
        raise ValueError(
            "No benchmark result files found."
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    return combined


def validate_experiments(meta_df):

    print("\n" + "-" * 75)
    print("VALIDATING EXPERIMENTS")
    print("-" * 75)

    expected_classifiers = set(
        CLASSIFIERS
    )

    expected_resampling_count = 7

    for dataset in PRIMARY_DATASETS:

        dataset_df = meta_df[
            meta_df["dataset"] == dataset
        ].copy()

        if len(dataset_df) != (
            EXPECTED_EXPERIMENTS_PER_DATASET
        ):
            raise ValueError(
                f"{dataset}: expected "
                f"{EXPECTED_EXPERIMENTS_PER_DATASET} "
                f"experiment rows, found "
                f"{len(dataset_df)}"
            )

        classifiers_found = set(
            dataset_df["classifier"].unique()
        )

        if classifiers_found != expected_classifiers:
            raise ValueError(
                f"{dataset}: unexpected classifiers: "
                f"{sorted(classifiers_found)}"
            )

        resampling_counts = (
            dataset_df
            .groupby("classifier")
            ["resampling_code"]
            .nunique()
        )

        for classifier in CLASSIFIERS:

            if classifier not in resampling_counts:
                raise ValueError(
                    f"{dataset}: missing classifier "
                    f"{classifier}"
                )

            if (
                resampling_counts[classifier]
                != expected_resampling_count
            ):
                raise ValueError(
                    f"{dataset} / {classifier}: expected "
                    f"{expected_resampling_count} "
                    f"resampling strategies, found "
                    f"{resampling_counts[classifier]}"
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
                f"{duplicates} duplicate experiment rows."
            )

        verification_values = set(
            dataset_df["verification"]
            .dropna()
            .astype(str)
        )

        invalid_statuses = (
            verification_values
            - {
                "PASS",
                "FAIL",
                "NOT_APPLICABLE",
            }
        )

        if invalid_statuses:
            raise ValueError(
                f"{dataset}: unexpected verification "
                f"status values: "
                f"{sorted(invalid_statuses)}"
            )

    expected_total = (
        EXPECTED_EXPERIMENTS_PER_DATASET
        * len(PRIMARY_DATASETS)
    )

    if len(meta_df) != expected_total:
        raise ValueError(
            f"Expected {expected_total} total experiment rows, "
            f"found {len(meta_df)}"
        )

    verification_failures = (
        meta_df["verification"]
        == "FAIL"
    ).sum()

    not_applicable = (
        meta_df["verification"]
        == "NOT_APPLICABLE"
    ).sum()

    successful = (
        meta_df["verification"]
        == "PASS"
    ).sum()

    print(
        f"\nTotal experiment rows : "
        f"{len(meta_df)}"
    )

    print(
        f"Successful rows       : "
        f"{successful}"
    )

    print(
        f"Verification failures : "
        f"{verification_failures}"
    )

    print(
        f"Not applicable        : "
        f"{not_applicable}"
    )

    if verification_failures != 0:

        failed_rows = meta_df[
            meta_df["verification"]
            == "FAIL"
        ]

        print(
            "\nFailed experiment rows:"
        )

        print(
            failed_rows[
                [
                    "dataset",
                    "classifier",
                    "resampling_code",
                    "verification",
                ]
            ].to_string(
                index=False
            )
        )

        raise ValueError(
            f"Found {verification_failures} "
            f"genuine verification failures."
        )

    return meta_df


def keep_model_ready_experiments(meta_df):

    print("\n" + "-" * 75)
    print("FILTERING MODEL-READY EXPERIMENTS")
    print("-" * 75)

    not_applicable = meta_df[
        meta_df["verification"]
        == "NOT_APPLICABLE"
    ].copy()

    if not_applicable.empty:
        print(
            "No NOT_APPLICABLE experiments found."
        )
    else:
        print(
            f"Excluding {len(not_applicable)} "
            f"NOT_APPLICABLE experiment rows."
        )

        print(
            "\nExcluded rows:"
        )

        print(
            not_applicable[
                [
                    "dataset",
                    "classifier",
                    "resampling_code",
                    "resampling",
                ]
            ].to_string(
                index=False
            )
        )

    model_ready = meta_df[
        meta_df["verification"]
        == "PASS"
    ].copy()

    if model_ready.empty:
        raise ValueError(
            "No model-ready experiments remain."
        )

    if model_ready[
        "macro_f1_math"
    ].isna().any():

        raise ValueError(
            "Model-ready experiments contain "
            "missing Macro-F1 targets."
        )

    print(
        f"\nModel-ready rows: "
        f"{len(model_ready)}"
    )

    return model_ready


def add_dataset_meta_features(meta_df):

    if not PROFILE_FILE.exists():
        raise FileNotFoundError(
            f"Dataset profile not found: "
            f"{PROFILE_FILE}"
        )

    profiles = pd.read_csv(
        PROFILE_FILE
    )

    print("\n" + "-" * 75)
    print("MERGING DATASET META-FEATURES")
    print("-" * 75)

    required_profile_columns = [
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
    ]

    missing_profile_columns = [
        column
        for column in required_profile_columns
        if column not in profiles.columns
    ]

    if missing_profile_columns:
        raise ValueError(
            "Dataset profile is missing columns: "
            f"{missing_profile_columns}"
        )

    duplicate_profiles = (
        profiles.duplicated(
            subset=["dataset"]
        ).sum()
    )

    if duplicate_profiles != 0:
        raise ValueError(
            f"Dataset profile contains "
            f"{duplicate_profiles} duplicate dataset rows."
        )

    profile_datasets = set(
        profiles["dataset"]
        .astype(str)
    )

    missing_datasets = (
        set(PRIMARY_DATASETS)
        - profile_datasets
    )

    if missing_datasets:
        raise ValueError(
            "Missing profile information for datasets: "
            f"{sorted(missing_datasets)}"
        )

    profiles = profiles[
        required_profile_columns
    ].copy()

    meta_df = meta_df.merge(
        profiles,
        on="dataset",
        how="left",
        suffixes=("", "_profile"),
    )

    for column in [
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
    ]:

        if column not in meta_df.columns:
            raise ValueError(
                f"Missing merged meta-feature: "
                f"{column}"
            )

        if meta_df[column].isna().any():
            raise ValueError(
                f"Missing meta-feature values in "
                f"column '{column}'."
            )

    print(
        f"Merged profile information for "
        f"{meta_df['dataset'].nunique()} datasets."
    )

    return meta_df


def add_meta_targets(meta_df):

    print("\n" + "-" * 75)
    print("CREATING META-LEARNING TARGETS")
    print("-" * 75)

    meta_df = meta_df.copy()

    meta_df["target_macro_f1"] = (
        meta_df["macro_f1_math"]
    )

    if meta_df[
        "target_macro_f1"
    ].isna().any():

        raise ValueError(
            "target_macro_f1 contains missing values."
        )

    baseline = meta_df[
        meta_df["resampling_code"]
        == "none"
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
        len(PRIMARY_DATASETS)
        * len(CLASSIFIERS)
    )

    if len(baseline) != expected_baselines:
        raise ValueError(
            f"Expected {expected_baselines} "
            f"baseline rows, found "
            f"{len(baseline)}"
        )

    duplicate_baselines = (
        baseline.duplicated(
            subset=[
                "dataset",
                "classifier",
            ]
        ).sum()
    )

    if duplicate_baselines != 0:
        raise ValueError(
            f"Found {duplicate_baselines} "
            f"duplicate baseline rows."
        )

    meta_df = meta_df.merge(
        baseline,
        on=[
            "dataset",
            "classifier",
        ],
        how="left",
    )

    if meta_df[
        "baseline_macro_f1"
    ].isna().any():

        raise ValueError(
            "Some experiments are missing "
            "their no-resampling baseline."
        )

    meta_df[
        "delta_macro_f1_vs_baseline"
    ] = (
        meta_df["target_macro_f1"]
        - meta_df["baseline_macro_f1"]
    )

    print(
        "Created target_macro_f1."
    )

    print(
        "Created baseline_macro_f1."
    )

    print(
        "Created delta_macro_f1_vs_baseline."
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
        "failure_reason",

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
            f"Missing expected output columns: "
            f"{missing_columns}"
        )

    return meta_df[
        final_columns
    ]


def validate_final_dataset(meta_df):

    print("\n" + "-" * 75)
    print("VALIDATING FINAL META-DATASET")
    print("-" * 75)

    expected_datasets = set(
        PRIMARY_DATASETS
    )

    actual_datasets = set(
        meta_df["dataset"]
        .astype(str)
        .unique()
    )

    if actual_datasets != expected_datasets:
        raise ValueError(
            "Final meta-dataset dataset set does "
            "not match the expected primary datasets.\n"
            f"Expected: {sorted(expected_datasets)}\n"
            f"Found   : {sorted(actual_datasets)}"
        )

    if meta_df[
        "target_macro_f1"
    ].isna().any():

        raise ValueError(
            "Final meta-dataset contains "
            "missing target values."
        )

    if meta_df[
        "baseline_macro_f1"
    ].isna().any():

        raise ValueError(
            "Final meta-dataset contains "
            "missing baseline values."
        )

    if meta_df[
        "delta_macro_f1_vs_baseline"
    ].isna().any():

        raise ValueError(
            "Final meta-dataset contains "
            "missing delta values."
        )

    remaining_failures = (
        meta_df["verification"]
        == "FAIL"
    ).sum()

    if remaining_failures != 0:
        raise ValueError(
            f"Final meta-dataset contains "
            f"{remaining_failures} FAIL rows."
        )

    duplicate_experiments = (
        meta_df.duplicated(
            subset=[
                "dataset",
                "resampling_code",
                "classifier",
            ]
        ).sum()
    )

    if duplicate_experiments != 0:
        raise ValueError(
            f"Final meta-dataset contains "
            f"{duplicate_experiments} duplicate "
            f"experiment combinations."
        )

    print(
        "Final meta-dataset validation: PASS"
    )


def print_summary(meta_df, original_df):

    print("\n" + "=" * 75)
    print("META-DATASET SUMMARY")
    print("=" * 75)

    print(
        f"Primary datasets          : "
        f"{len(PRIMARY_DATASETS)}"
    )

    print(
        f"Original experiment rows  : "
        f"{len(original_df)}"
    )

    print(
        f"Model-ready experiment rows: "
        f"{len(meta_df)}"
    )

    print(
        f"Datasets in meta-dataset  : "
        f"{meta_df['dataset'].nunique()}"
    )

    print(
        f"Resampling methods        : "
        f"{meta_df['resampling_code'].nunique()}"
    )

    print(
        f"Classifiers               : "
        f"{meta_df['classifier'].nunique()}"
    )

    print("\nExperiments per dataset:")

    print(
        meta_df[
            "dataset"
        ].value_counts()
        .sort_index()
    )

    print("\nExperiments per classifier:")

    print(
        meta_df[
            "classifier"
        ].value_counts()
        .sort_index()
    )

    print("\nValid Macro-F1 range:")

    print(
        f"Minimum: "
        f"{meta_df['target_macro_f1'].min():.6f}"
    )

    print(
        f"Maximum: "
        f"{meta_df['target_macro_f1'].max():.6f}"
    )

    print("\nMacro-F1 target statistics:")

    print(
        meta_df[
            "target_macro_f1"
        ].describe()
        .to_string()
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
                "baseline_macro_f1",
                "delta_macro_f1_vs_baseline",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )


def main():

    print("=" * 75)
    print(
        "ResampleRank - Building Primary Meta-Dataset"
    )
    print("=" * 75)

    meta_df = load_result_files()

    validated_df = validate_experiments(
        meta_df
    )

    model_ready_df = (
        keep_model_ready_experiments(
            validated_df
        )
    )

    model_ready_df = (
        add_dataset_meta_features(
            model_ready_df
        )
    )

    model_ready_df = (
        add_meta_targets(
            model_ready_df
        )
    )

    model_ready_df = (
        select_columns(
            model_ready_df
        )
    )

    model_ready_df = (
        model_ready_df
        .sort_values(
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
        )
        .reset_index(
            drop=True
        )
    )

    validate_final_dataset(
        model_ready_df
    )

    model_ready_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print_summary(
        model_ready_df,
        meta_df,
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()