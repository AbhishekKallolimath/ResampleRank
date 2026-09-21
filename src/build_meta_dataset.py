from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = BASE_DIR / "results"
PROFILE_FILE = BASE_DIR / "data" / "processed" / "dataset_profiles.csv"

OUTPUT_FILE = RESULTS_DIR / "pima_meta_dataset.csv"


# ---------------------------------------------------------
# Input result files
# ---------------------------------------------------------
RESULT_FILES = [
    RESULTS_DIR / "pima_resampling_results.csv",
    RESULTS_DIR / "pima_decision_tree_results.csv",
    RESULTS_DIR / "pima_random_forest_results.csv",
    RESULTS_DIR / "pima_xgboost_results.csv",
]


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    print("=" * 75)
    print("ResampleRank - Building Pima Meta-Dataset")
    print("=" * 75)

    frames = []

    for file_path in RESULT_FILES:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required result file not found: {file_path}"
            )

        df = pd.read_csv(file_path)

        print(
            f"\nLoaded: {file_path.name}"
            f" | Rows: {len(df)}"
        )

        frames.append(df)

    # -----------------------------------------------------
    # Combine all experiments
    # -----------------------------------------------------
    meta_df = pd.concat(
        frames,
        ignore_index=True,
    )

    # -----------------------------------------------------
    # Basic validation
    # -----------------------------------------------------
    expected_rows = 28

    if len(meta_df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} experiments, "
            f"but found {len(meta_df)}."
        )

    duplicates = meta_df.duplicated(
        subset=[
            "dataset",
            "resampling_code",
            "classifier",
        ]
    ).sum()

    if duplicates != 0:
        raise ValueError(
            f"Found {duplicates} duplicate experiment rows."
        )

    if not (
        meta_df["verification"] == "PASS"
    ).all():
        raise ValueError(
            "At least one experiment failed mathematical verification."
        )

    # -----------------------------------------------------
    # Load dataset meta-features
    # -----------------------------------------------------
    if not PROFILE_FILE.exists():
        raise FileNotFoundError(
            f"Dataset profile not found: {PROFILE_FILE}"
        )

    profiles = pd.read_csv(PROFILE_FILE)

    if "dataset" not in profiles.columns:
        raise ValueError(
            "Dataset profile must contain a 'dataset' column."
        )

    # Keep one profile per dataset
    profiles = profiles.drop_duplicates(
        subset=["dataset"]
    )

    # -----------------------------------------------------
    # Merge experimental results with dataset characteristics
    # -----------------------------------------------------
    meta_df = meta_df.merge(
        profiles,
        on="dataset",
        how="left",
        suffixes=("", "_profile"),
    )

    # -----------------------------------------------------
    # Verify merge
    # -----------------------------------------------------
    profile_columns = [
        "samples",
        "features",
        "imbalance_ratio",
        "minority_proportion",
    ]

    for column in profile_columns:
        if column not in meta_df.columns:
            raise ValueError(
                f"Missing required meta-feature column: {column}"
            )

    # -----------------------------------------------------
    # Create research target
    # -----------------------------------------------------
    meta_df["target_macro_f1"] = meta_df[
        "macro_f1_math"
    ]

    # -----------------------------------------------------
    # Calculate improvement over no-resampling
    # for the same dataset + classifier
    # -----------------------------------------------------
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
            "target_macro_f1": "baseline_macro_f1"
        }
    )

    meta_df = meta_df.merge(
        baseline,
        on=["dataset", "classifier"],
        how="left",
    )

    meta_df["delta_macro_f1_vs_baseline"] = (
        meta_df["target_macro_f1"]
        - meta_df["baseline_macro_f1"]
    )

    # -----------------------------------------------------
    # Select and order useful columns
    # -----------------------------------------------------
    final_columns = [
        # Dataset
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

        # Experimental configuration
        "resampling_code",
        "resampling",
        "classifier",

        # Confusion matrix
        "tn",
        "fp",
        "fn",
        "tp",

        # Metrics
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1_class_0_math",
        "f1_class_1_math",
        "macro_f1_code",
        "macro_f1_math",

        # Verification
        "metric_error",
        "verification",

        # Runtime
        "execution_time_seconds",

        # Meta-learning target
        "target_macro_f1",

        # Baseline comparison
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
            f"Missing expected columns: {missing_columns}"
        )

    meta_df = meta_df[final_columns]

    # -----------------------------------------------------
    # Sort consistently
    # -----------------------------------------------------
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
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------
    meta_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------
    print("\n" + "-" * 75)
    print("META-DATASET SUMMARY")
    print("-" * 75)

    print(f"Total experiments : {len(meta_df)}")
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
        f"\nMathematical verification failures: "
        f"{(meta_df['verification'] != 'PASS').sum()}"
    )

    print("\nMacro-F1 range:")
    print(
        f"Minimum: {meta_df['target_macro_f1'].min():.6f}"
    )
    print(
        f"Maximum: {meta_df['target_macro_f1'].max():.6f}"
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
                "execution_time_seconds",
            ]
        ].to_string(index=False)
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()