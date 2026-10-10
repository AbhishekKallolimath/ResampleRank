from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import NearestNeighbors

import run_registry_benchmark as registry_loader


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
REGISTRY_FILE = BASE_DIR / "data" / "dataset_registry.csv"
OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "dataset_profiles.csv"
)

VALID_SPLITS = {
    "TRAIN",
    "VALIDATION",
    "TEST",
}

EXPECTED_EXPERIMENTS = 28

REQUIRED_RESULT_COLUMNS = [
    "dataset",
    "classifier",
    "resampling_code",
    "macro_f1_math",
    "verification",
]


def read_registry():

    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(
            "Registry not found: {}".format(
                REGISTRY_FILE
            )
        )

    registry = pd.read_csv(
        REGISTRY_FILE
    )

    required = [
        "dataset",
        "source_file",
        "split",
    ]

    missing = [
        column
        for column in required
        if column not in registry.columns
    ]

    if missing:
        raise ValueError(
            "Missing registry columns: {}".format(
                missing
            )
        )

    registry["split"] = (
        registry["split"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return registry


def load_dataset_results(dataset_name):

    frames = []

    for file_path in RESULTS_DIR.glob(
        "{}*_results.csv".format(
            dataset_name
        )
    ):

        if file_path.name.endswith(
            "_all_results.csv"
        ):
            continue

        try:
            frame = pd.read_csv(
                file_path
            )
        except Exception:
            continue

        if not all(
            column in frame.columns
            for column in REQUIRED_RESULT_COLUMNS
        ):
            continue

        frame = frame[
            frame["dataset"].astype(str)
            == dataset_name
        ].copy()

        if not frame.empty:
            frames.append(
                frame
            )

    if not frames:
        return pd.DataFrame(
            columns=REQUIRED_RESULT_COLUMNS
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=[
            "dataset",
            "classifier",
            "resampling_code",
        ],
        keep="first",
    )

    return combined


def is_clean_dataset(dataset_name):

    results = load_dataset_results(
        dataset_name
    )

    if results.empty:
        return False, "NO_RESULTS"

    if len(results) != EXPECTED_EXPERIMENTS:
        return (
            False,
            "EXPECTED_28_FOUND_{}".format(
                len(results)
            ),
        )

    verification = (
        results["verification"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if not verification.eq("PASS").all():
        return False, "VERIFICATION_FAILURE"

    if results["macro_f1_math"].isna().any():
        return False, "N/A_OR_MISSING_MACRO_F1"

    classifiers = (
        results["classifier"]
        .astype(str)
        .str.strip()
        .nunique()
    )

    resampling_codes = (
        results["resampling_code"]
        .astype(str)
        .str.strip()
        .nunique()
    )

    if classifiers != 4:
        return (
            False,
            "CLASSIFIERS_FOUND_{}".format(
                classifiers
            ),
        )

    if resampling_codes != 7:
        return (
            False,
            "RESAMPLING_STRATEGIES_FOUND_{}".format(
                resampling_codes
            ),
        )

    return True, "CLEAN"


def calculate_class_entropy(y):

    probabilities = (
        y.value_counts(normalize=True)
        .to_numpy(dtype=float)
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    return float(
        -np.sum(
            probabilities
            * np.log2(probabilities)
        )
    )


def calculate_numeric_statistics(
    X,
    numeric_features,
):

    if not numeric_features:
        return {
            "mean_feature_variance": 0.0,
            "mean_feature_std": 0.0,
            "mean_abs_feature_skewness": 0.0,
        }

    numeric_df = X[
        numeric_features
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    variances = numeric_df.var(
        ddof=0
    )

    stds = numeric_df.std(
        ddof=0
    )

    skewness = numeric_df.skew(
        skipna=True
    )

    return {
        "mean_feature_variance": float(
            variances.mean()
        ),
        "mean_feature_std": float(
            stds.mean()
        ),
        "mean_abs_feature_skewness": float(
            skewness.abs().mean()
        ),
    }


def calculate_correlation_features(
    X,
    numeric_features,
):

    if len(numeric_features) < 2:
        return {
            "mean_abs_correlation": 0.0,
            "max_abs_correlation": 0.0,
        }

    numeric_df = X[
        numeric_features
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    correlation = (
        numeric_df
        .corr()
        .abs()
        .fillna(0.0)
    )

    values = correlation.to_numpy()

    upper_indices = np.triu_indices(
        values.shape[0],
        k=1,
    )

    upper_values = values[
        upper_indices
    ]

    if len(upper_values) == 0:
        return {
            "mean_abs_correlation": 0.0,
            "max_abs_correlation": 0.0,
        }

    return {
        "mean_abs_correlation": float(
            upper_values.mean()
        ),
        "max_abs_correlation": float(
            upper_values.max()
        ),
    }


def calculate_zero_proportion(
    X,
    numeric_features,
):

    if not numeric_features:
        return 0.0

    numeric_df = X[
        numeric_features
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    zero_proportions = (
        (numeric_df == 0)
        .mean()
    )

    return float(
        zero_proportions.mean()
    )


def calculate_unique_ratio(
    X,
):

    if X.shape[1] == 0:
        return 0.0

    ratios = []

    for column in X.columns:
        ratios.append(
            X[column]
            .nunique(dropna=False)
            / float(len(X))
        )

    return float(
        np.mean(ratios)
    )


def calculate_mutual_information(
    X,
    y,
    numeric_features,
):

    encoded_columns = []
    discrete_flags = []

    for column in X.columns:

        if column in numeric_features:

            values = (
                pd.to_numeric(
                    X[column],
                    errors="coerce",
                )
                .to_numpy(
                    dtype=float
                )
            )

            if not np.isfinite(
                values
            ).all():

                raise ValueError(
                    "Non-finite numeric values "
                    "found in '{}'.".format(
                        column
                    )
                )

            if np.std(values) == 0:
                values = np.zeros(
                    len(values),
                    dtype=float,
                )

            encoded_columns.append(
                values
            )

            discrete_flags.append(
                False
            )

        else:

            codes, _ = pd.factorize(
                X[column]
                .astype(str)
                .str.strip()
            )

            encoded_columns.append(
                codes.astype(float)
            )

            discrete_flags.append(
                True
            )

    if not encoded_columns:
        return {
            "mean_mutual_information": 0.0,
            "max_mutual_information": 0.0,
        }

    encoded = np.column_stack(
        encoded_columns
    )

    y_array = np.asarray(
        y,
        dtype=int,
    )

    mutual_information = (
        mutual_info_classif(
            encoded,
            y_array,
            discrete_features=np.asarray(
                discrete_flags,
                dtype=bool,
            ),
            random_state=42,
        )
    )

    mutual_information = np.nan_to_num(
        mutual_information,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return {
        "mean_mutual_information": float(
            mutual_information.mean()
        ),
        "max_mutual_information": float(
            mutual_information.max()
        ),
    }


def encode_for_nearest_neighbors(
    X,
    numeric_features,
):

    parts = []

    for column in X.columns:

        if column in numeric_features:

            values = (
                pd.to_numeric(
                    X[column],
                    errors="coerce",
                )
                .to_numpy(
                    dtype=float
                )
            )

            mean = values.mean()
            std = values.std()

            if std > 0:
                values = (
                    values - mean
                ) / std
            else:
                values = np.zeros(
                    len(values),
                    dtype=float,
                )

            parts.append(
                values.reshape(
                    -1,
                    1,
                )
            )

        else:

            dummy = pd.get_dummies(
                X[column]
                .astype(str)
                .str.strip(),
                prefix=column,
            )

            parts.append(
                dummy.to_numpy(
                    dtype=float
                )
            )

    if not parts:

        return np.zeros(
            (
                len(X),
                1,
            ),
            dtype=float,
        )

    return np.hstack(
        parts
    )


def calculate_nearest_neighbor_features(
    X,
    y,
    numeric_features,
):

    n_samples = len(X)

    if n_samples < 3:
        return {
            "one_nn_error_rate": 0.0,
            "minority_majority_neighbor_ratio": 0.0,
            "minority_mean_neighbor_distance": 0.0,
        }

    encoded = (
        encode_for_nearest_neighbors(
            X,
            numeric_features,
        )
    )

    y_array = np.asarray(
        y,
        dtype=int,
    )

    k = min(
        5,
        n_samples - 1,
    )

    neighbors = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
        n_jobs=1,
    )

    neighbors.fit(
        encoded
    )

    distances, indices = (
        neighbors.kneighbors(
            encoded,
            return_distance=True,
        )
    )

    neighbor_distances = (
        distances[:, 1:]
    )

    neighbor_indices = (
        indices[:, 1:]
    )

    nearest_neighbor_labels = (
        y_array[
            neighbor_indices[:, 0]
        ]
    )

    one_nn_error_rate = float(
        np.mean(
            nearest_neighbor_labels
            != y_array
        )
    )

    minority_mask = (
        y_array == 1
    )

    if minority_mask.sum() == 0:
        minority_majority_neighbor_ratio = 0.0
        minority_mean_neighbor_distance = 0.0

    else:

        minority_neighbor_indices = (
            neighbor_indices[
                minority_mask
            ]
        )

        minority_neighbor_distances = (
            neighbor_distances[
                minority_mask
            ]
        )

        minority_neighbor_labels = (
            y_array[
                minority_neighbor_indices
            ]
        )

        minority_majority_neighbor_ratio = float(
            np.mean(
                minority_neighbor_labels
                == 0
            )
        )

        minority_mean_neighbor_distance = float(
            np.mean(
                minority_neighbor_distances
            )
        )

    return {
        "one_nn_error_rate": one_nn_error_rate,
        "minority_majority_neighbor_ratio":
            minority_majority_neighbor_ratio,
        "minority_mean_neighbor_distance":
            minority_mean_neighbor_distance,
    }


def build_profile(
    dataset_name,
    split,
):

    (
        X,
        y,
        majority_class,
        minority_class,
        numeric_features,
        categorical_features,
    ) = registry_loader.load_registry_dataset(
        dataset_name
    )

    class_counts = y.value_counts()

    majority_count = int(
        class_counts.max()
    )

    minority_count = int(
        class_counts.min()
    )

    imbalance_ratio = (
        majority_count
        / minority_count
    )

    numeric_statistics = (
        calculate_numeric_statistics(
            X,
            numeric_features,
        )
    )

    correlation_features = (
        calculate_correlation_features(
            X,
            numeric_features,
        )
    )

    zero_proportion = (
        calculate_zero_proportion(
            X,
            numeric_features,
        )
    )

    unique_ratio = (
        calculate_unique_ratio(
            X,
        )
    )

    class_entropy = (
        calculate_class_entropy(
            y,
        )
    )

    mutual_information = (
        calculate_mutual_information(
            X,
            y,
            numeric_features,
        )
    )

    nearest_neighbor_features = (
        calculate_nearest_neighbor_features(
            X,
            y,
            numeric_features,
        )
    )

    profile = {
        "dataset": dataset_name,
        "split": split,

        # Basic dataset structure
        "samples": int(
            X.shape[0]
        ),
        "features": int(
            X.shape[1]
        ),
        "feature_sample_ratio": round(
            X.shape[1]
            / X.shape[0],
            6,
        ),
        "numeric_features": int(
            len(numeric_features)
        ),
        "categorical_features": int(
            len(categorical_features)
        ),

        # Data quality
        "missing_values": int(
            X.isnull().sum().sum()
        ),
        "duplicates": int(
            X.duplicated().sum()
        ),

        # Class imbalance
        "majority_class": majority_class,
        "minority_class": minority_class,
        "majority_count": majority_count,
        "minority_count": minority_count,
        "minority_proportion": round(
            minority_count
            / len(y),
            6,
        ),
        "imbalance_ratio": round(
            imbalance_ratio,
            6,
        ),
        "class_entropy": round(
            class_entropy,
            6,
        ),

        # Feature distribution
        "mean_feature_variance": round(
            numeric_statistics[
                "mean_feature_variance"
            ],
            6,
        ),
        "mean_feature_std": round(
            numeric_statistics[
                "mean_feature_std"
            ],
            6,
        ),
        "mean_abs_feature_skewness": round(
            numeric_statistics[
                "mean_abs_feature_skewness"
            ],
            6,
        ),

        # Feature structure
        "mean_unique_value_ratio": round(
            unique_ratio,
            6,
        ),
        "mean_abs_correlation": round(
            correlation_features[
                "mean_abs_correlation"
            ],
            6,
        ),
        "max_abs_correlation": round(
            correlation_features[
                "max_abs_correlation"
            ],
            6,
        ),
        "mean_zero_proportion": round(
            zero_proportion,
            6,
        ),

        # Feature-label relationship
        "mean_mutual_information": round(
            mutual_information[
                "mean_mutual_information"
            ],
            6,
        ),
        "max_mutual_information": round(
            mutual_information[
                "max_mutual_information"
            ],
            6,
        ),

        # Complexity / overlap
        "one_nn_error_rate": round(
            nearest_neighbor_features[
                "one_nn_error_rate"
            ],
            6,
        ),

        # Minority-class difficulty
        "minority_majority_neighbor_ratio":
            round(
                nearest_neighbor_features[
                    "minority_majority_neighbor_ratio"
                ],
                6,
            ),

        "minority_mean_neighbor_distance":
            round(
                nearest_neighbor_features[
                    "minority_mean_neighbor_distance"
                ],
                6,
            ),
    }

    return profile


def main():

    registry = read_registry()

    candidates = registry[
        registry["split"].isin(
            VALID_SPLITS
        )
        & registry["source_file"].notna()
        & registry[
            "source_file"
        ].astype(str).str.strip().ne("")
    ].copy()

    candidates = candidates.drop_duplicates(
        subset=["dataset"]
    )

    clean_datasets = []
    excluded = []

    for _, row in candidates.iterrows():

        dataset_name = str(
            row["dataset"]
        ).strip()

        split = str(
            row["split"]
        ).strip().upper()

        try:
            clean, reason = (
                is_clean_dataset(
                    dataset_name
                )
            )

        except Exception as exc:

            clean = False

            reason = (
                "CHECK_ERROR: {}".format(
                    exc
                )
            )

        if clean:

            clean_datasets.append(
                (
                    dataset_name,
                    split,
                )
            )

        else:

            excluded.append(
                (
                    dataset_name,
                    split,
                    reason,
                )
            )

    print("=" * 75)
    print(
        "ResampleRank - Registry Dataset Profiler"
    )
    print("=" * 75)

    print(
        "\nRegistry candidates : {}".format(
            len(candidates)
        )
    )

    print(
        "Clean datasets      : {}".format(
            len(clean_datasets)
        )
    )

    split_counts = {}

    for _, split in clean_datasets:

        split_counts[split] = (
            split_counts.get(
                split,
                0,
            )
            + 1
        )

    print(
        "\nClean split counts:"
    )

    for split in [
        "TRAIN",
        "VALIDATION",
        "TEST",
    ]:

        print(
            "  {:12s}: {}".format(
                split,
                split_counts.get(
                    split,
                    0,
                ),
            )
        )

    if excluded:

        print(
            "\nExcluded datasets:"
        )

        for (
            dataset_name,
            split,
            reason,
        ) in excluded:

            print(
                "  {} [{}] -> {}".format(
                    dataset_name,
                    split,
                    reason,
                )
            )

    profiles = []

    for dataset_name, split in clean_datasets:

        try:

            profile = build_profile(
                dataset_name,
                split,
            )

            profiles.append(
                profile
            )

            print(
                "\nPROFILE PASS: {}".format(
                    dataset_name
                )
            )

        except Exception as exc:

            print(
                "\nPROFILE FAIL: {} -> {}".format(
                    dataset_name,
                    exc,
                )
            )

    profile_df = pd.DataFrame(
        profiles
    )

    if profile_df.empty:

        raise RuntimeError(
            "No dataset profiles were created."
        )

    split_order = {
        "TRAIN": 0,
        "VALIDATION": 1,
        "TEST": 2,
    }

    profile_df["_split_order"] = (
        profile_df["split"]
        .map(split_order)
    )

    profile_df = (
        profile_df
        .sort_values(
            [
                "_split_order",
                "dataset",
            ]
        )
        .drop(
            columns=[
                "_split_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\nSaved profiles: {}".format(
            OUTPUT_FILE
        )
    )

    print(
        "\nProfile columns:"
    )

    for column in profile_df.columns:
        print(
            "  {}".format(
                column
            )
        )

    print(
        "\nFinal profile counts:"
    )

    print(
        profile_df["split"]
        .value_counts()
        .reindex(
            [
                "TRAIN",
                "VALIDATION",
                "TEST",
            ],
            fill_value=0,
        )
    )

    print(
        "\nTotal profiles: {}".format(
            len(profile_df)
        )
    )


if __name__ == "__main__":
    main()