from pathlib import Path
import math
import sys

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROFILE_FILE = (
    PROCESSED_DIR / "dataset_profiles_clean43.csv"
)
SPLIT_FILE = (
    BASE_DIR / "data" / "dataset_splits_clean43.csv"
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_registry_benchmark as benchmark


SPLITS = {
    "TRAIN": [
        "glass0",
        "glass1",
        "glass2",
        "glass4",
        "glass5",
        "glass6",
        "glass-0-1-5_vs_2",
        "glass-0-1-6_vs_5",
        "vehicle0",
        "vehicle1",
        "vehicle2",
        "vehicle3",
        "yeast1",
        "yeast4",
        "yeast5",
        "yeast6",
        "yeast-0-5-6-7-9_vs_4",
        "yeast-1-2-8-9_vs_7",
        "yeast-1-4-5-8_vs_7",
        "yeast-1_vs_7",
        "yeast-2_vs_4",
        "yeast-2_vs_8",
        "IBM Employee Attrition",
    ],
    "VALIDATION": [
        "ecoli-0_vs_1",
        "ecoli1",
        "ecoli2",
        "ecoli3",
        "ecoli4",
        "ecoli-0-3-4_vs_5",
        "ecoli-0-6-7_vs_3-5",
        "ecoli-0-2-3-4_vs_5",
        "ecoli-0-1-3-7_vs_2-6",
        "page-blocks0",
        "abalone19",
        "abalone9-18",
        "new-thyroid1",
        "new-thyroid2",
    ],
    "TEST": [
        "pima",
        "wisconsin",
        "haberman",
        "segment0",
        "vowel0",
        "zoo-3",
    ],
}


def profile_dataset(dataset_name):
    (
        X,
        y,
        majority_class,
        minority_class,
        numeric_features,
        categorical_features,
    ) = benchmark.load_registry_dataset(dataset_name)

    if len(X) != len(y):
        raise ValueError("Feature/target length mismatch.")

    if y.nunique() != 2:
        raise ValueError(
            "Expected two target classes; found {}.".format(
                y.nunique()
            )
        )

    counts = y.value_counts()
    majority_count = int(counts.max())
    minority_count = int(counts.min())

    minority_proportion = minority_count / len(y)
    imbalance_ratio = majority_count / minority_count

    p_majority = majority_count / len(y)
    p_minority = minority_count / len(y)

    class_entropy = -(
        p_majority * math.log(p_majority, 2)
        + p_minority * math.log(p_minority, 2)
    )

    numeric_std = pd.Series(dtype=float)
    numeric_abs_skew = pd.Series(dtype=float)
    numeric_abs_kurtosis = pd.Series(dtype=float)

    if numeric_features:
        numeric_data = X[numeric_features].apply(
            pd.to_numeric,
            errors="raise",
        )
        numeric_std = numeric_data.std(ddof=0)
        numeric_abs_skew = numeric_data.skew().abs()
        numeric_abs_kurtosis = numeric_data.kurtosis().abs()

    categorical_cardinality = pd.Series(dtype=float)

    if categorical_features:
        categorical_cardinality = (
            X[categorical_features]
            .nunique(dropna=False)
            .astype(float)
        )

    return {
        "dataset": dataset_name,
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
        "feature_sample_ratio": round(
            X.shape[1] / X.shape[0], 6
        ),
        "numeric_features": int(len(numeric_features)),
        "categorical_features": int(len(categorical_features)),
        "missing_values": int(X.isnull().sum().sum()),
        "duplicates": int(X.duplicated().sum()),
        "majority_class": str(majority_class),
        "minority_class": str(minority_class),
        "majority_count": majority_count,
        "minority_count": minority_count,
        "minority_proportion": round(
            minority_proportion, 6
        ),
        "imbalance_ratio": round(imbalance_ratio, 6),
        "class_entropy": round(class_entropy, 6),
        "normalized_class_entropy": round(
            class_entropy, 6
        ),
        "constant_features": int(
            sum(X[col].nunique(dropna=False) <= 1
                for col in X.columns)
        ),
        "numeric_mean_std": (
            round(float(numeric_std.mean()), 6)
            if not numeric_std.empty else None
        ),
        "numeric_median_std": (
            round(float(numeric_std.median()), 6)
            if not numeric_std.empty else None
        ),
        "numeric_mean_abs_skew": (
            round(float(numeric_abs_skew.mean()), 6)
            if not numeric_abs_skew.empty
            and numeric_abs_skew.notna().any()
            else None
        ),
        "numeric_mean_abs_kurtosis": (
            round(float(numeric_abs_kurtosis.mean()), 6)
            if not numeric_abs_kurtosis.empty
            and numeric_abs_kurtosis.notna().any()
            else None
        ),
        "categorical_mean_cardinality": (
            round(float(categorical_cardinality.mean()), 6)
            if not categorical_cardinality.empty else None
        ),
    }


def main():
    expected_count = sum(
        len(datasets) for datasets in SPLITS.values()
    )

    all_datasets = [
        dataset
        for datasets in SPLITS.values()
        for dataset in datasets
    ]

    if len(all_datasets) != expected_count:
        raise ValueError("Unexpected split-list count.")

    if len(set(all_datasets)) != len(all_datasets):
        raise ValueError("A dataset appears in multiple splits.")

    profiles = []
    failures = []

    print("ResampleRank - Clean Dataset Profiling")
    print("=" * 55)
    print("Expected datasets:", expected_count)

    for split_name, datasets in SPLITS.items():
        print("{} datasets: {}".format(
            split_name, len(datasets)
        ))

    for index, dataset_name in enumerate(
        all_datasets, start=1
    ):
        print(
            "\n[{}/{}] Profiling {}".format(
                index, expected_count, dataset_name
            )
        )

        try:
            profiles.append(
                profile_dataset(dataset_name)
            )
            print("  PASS")
        except Exception as exc:
            failures.append((dataset_name, str(exc)))
            print("  FAIL: {}".format(exc))

    if failures:
        print("\nProfiling failed for {} dataset(s).".format(
            len(failures)
        ))
        for dataset_name, error in failures:
            print("  {}: {}".format(dataset_name, error))

        print("No output files were written.")
        raise SystemExit(1)

    profiles_df = pd.DataFrame(profiles)

    splits_df = pd.DataFrame([
        {"dataset": dataset, "split": split_name}
        for split_name, datasets in SPLITS.items()
        for dataset in datasets
    ])

    if len(profiles_df) != 43 or profiles_df["dataset"].nunique() != 43:
        raise ValueError(
            "Profile validation failed: expected 43 unique datasets."
        )

    if len(splits_df) != 43 or splits_df["dataset"].nunique() != 43:
        raise ValueError(
            "Split validation failed: expected 43 unique datasets."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    profiles_df.to_csv(PROFILE_FILE, index=False)
    splits_df.to_csv(SPLIT_FILE, index=False)

    print("\n" + "=" * 55)
    print("PROFILE GENERATION PASS")
    print("Profile rows:", len(profiles_df))
    print("Unique datasets:", profiles_df["dataset"].nunique())
    print("Split counts:")
    print(splits_df["split"].value_counts().to_string())
    print("\nProfile file:", PROFILE_FILE)
    print("Split file:", SPLIT_FILE)
    print("\nProfile preview:")
    print(
        profiles_df[
            [
                "dataset",
                "samples",
                "features",
                "imbalance_ratio",
                "numeric_features",
                "categorical_features",
            ]
        ].head(5).to_string(index=False)
    )


if __name__ == "__main__":
    main()
