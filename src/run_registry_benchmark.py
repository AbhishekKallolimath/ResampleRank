
from pathlib import Path
import argparse
import io
import zipfile

import pandas as pd

import run_full_benchmark as benchmark


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
REGISTRY_FILE = BASE_DIR / "data" / "dataset_registry.csv"


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
        "target_file",
        "loader_type",
        "separator",
        "header_row",
        "target_column",
        "target_column_index",
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

    return registry


def get_registry_row(
    registry,
    dataset_name,
):
    matches = registry[
        registry["dataset"].astype(str)
        == dataset_name
    ]

    if matches.empty:
        raise ValueError(
            "Dataset '{}' not found in registry.".format(
                dataset_name
            )
        )

    return matches.iloc[0]


def normalize_separator(value):
    if pd.isna(value):
        return ","

    value = str(value)

    if value == "nan":
        return ","

    return value


def parse_target_labels(y):
    y = (
        y.astype(str)
        .str.strip()
    )

    if y.isna().any():
        raise ValueError(
            "Target contains missing values."
        )

    counts = y.value_counts()

    if len(counts) != 2:
        raise ValueError(
            "Expected binary classification, found "
            "{} classes: {}".format(
                len(counts),
                counts.to_dict(),
            )
        )

    majority_class = counts.idxmax()
    minority_class = counts.idxmin()

    mapping = {
        majority_class: 0,
        minority_class: 1,
    }

    y_encoded = (
        y.map(mapping)
        .astype(int)
    )

    return (
        y_encoded,
        majority_class,
        minority_class,
    )


def parse_keel_text(
    content,
    dataset_name,
):
    text = content.decode(
        "utf-8",
        errors="replace",
    )

    lines = text.splitlines()

    attributes = []
    data_start = None

    for index, raw_line in enumerate(lines):
        line = raw_line.strip()

        if not line:
            continue

        if line.lower().startswith(
            "@attribute"
        ):
            parts = line.split(
                maxsplit=2
            )

            if len(parts) < 3:
                raise ValueError(
                    "Invalid KEEL attribute line: {}".format(
                        line
                    )
                )

            name = parts[1].strip()
            definition = parts[2].strip()

            if (
                definition.startswith("{")
                and definition.endswith("}")
            ):
                feature_type = "categorical"
            elif definition.lower().startswith(
                ("real", "integer")
            ):
                feature_type = "numeric"
            else:
                raise ValueError(
                    "Unsupported KEEL attribute definition "
                    "for '{}': {}".format(
                        name,
                        definition,
                    )
                )

            attributes.append(
                (
                    name,
                    feature_type,
                )
            )

        elif line.lower() == "@data":
            data_start = index + 1
            break

    if not attributes:
        raise ValueError(
            "No KEEL attributes found for '{}'.".format(
                dataset_name
            )
        )

    if data_start is None:
        raise ValueError(
            "KEEL @data marker not found for '{}'.".format(
                dataset_name
            )
        )

    data_rows = []

    for raw_line in lines[data_start:]:
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith("%"):
            continue

        values = [
            value.strip()
            for value in line.split(",")
        ]

        if len(values) != len(attributes):
            raise ValueError(
                "KEEL data row has {} values; "
                "expected {} in '{}'.".format(
                    len(values),
                    len(attributes),
                    dataset_name,
                )
            )

        data_rows.append(values)

    if not data_rows:
        raise ValueError(
            "No KEEL data rows found for '{}'.".format(
                dataset_name
            )
        )

    columns = [
        name
        for name, _ in attributes
    ]

    X = pd.DataFrame(
        data_rows,
        columns=columns,
    )

    feature_types = {
        name: feature_type
        for name, feature_type
        in attributes
    }

    return X, feature_types


def load_keel_source(
    source_path,
    dataset_name,
):
    if source_path.suffix.lower() == ".dat":
        with open(
            source_path,
            "rb",
        ) as file:
            content = file.read()

        return parse_keel_text(
            content,
            dataset_name,
        )

    if source_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(
            source_path,
            "r",
        ) as archive:

            entries = [
                name
                for name in archive.namelist()
                if (
                    name.lower().endswith(
                        ".dat"
                    )
                    and "__MACOSX"
                    not in name
                    and not Path(
                        name
                    ).name.startswith(
                        "._"
                    )
                )
            ]

            if not entries:
                raise ValueError(
                    "No .dat file found in ZIP: {}".format(
                        source_path
                    )
                )

            dat_name = entries[0]

            with archive.open(
                dat_name,
                "r",
            ) as file:
                content = file.read()

        return parse_keel_text(
            content,
            dataset_name,
        )

    raise ValueError(
        "Unsupported KEEL source: {}".format(
            source_path
        )
    )


def prepare_keel_dataset(
    row,
    dataset_name,
):
    source_file = str(
        row["source_file"]
    ).strip()

    source_path = RAW_DIR / source_file

    X, feature_types = load_keel_source(
        source_path,
        dataset_name,
    )

    target_value = row[
        "target_column"
    ]

    if pd.isna(target_value):
        raise ValueError(
            "target_column is empty for '{}'.".format(
                dataset_name
            )
        )

    target_column = str(
        target_value
    ).strip()

    if target_column not in X.columns:
        raise ValueError(
            "Target '{}' not found in KEEL dataset '{}'. "
            "Columns: {}".format(
                target_column,
                dataset_name,
                list(X.columns),
            )
        )

    y_raw = X[target_column].copy()

    X = X.drop(
        columns=[target_column]
    )

    numeric_features = [
        column
        for column in X.columns
        if feature_types[column]
        == "numeric"
    ]

    categorical_features = [
        column
        for column in X.columns
        if feature_types[column]
        == "categorical"
    ]

    for column in numeric_features:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    for column in categorical_features:
        X[column] = (
            X[column]
            .astype(str)
            .str.strip()
        )

    if X.isnull().any().any():
        missing = X.columns[
            X.isnull().any()
        ].tolist()

        raise ValueError(
            "Missing/invalid feature values in '{}': {}".format(
                dataset_name,
                missing,
            )
        )

    y, majority_class, minority_class = (
        parse_target_labels(y_raw)
    )

    return (
        X,
        y,
        majority_class,
        minority_class,
        numeric_features,
        categorical_features,
    )


def prepare_csv_dataset(
    row,
    dataset_name,
):
    source_file = str(
        row["source_file"]
    ).strip()

    source_path = RAW_DIR / source_file

    separator = normalize_separator(
        row["separator"]
    )

    header_value = row[
        "header_row"
    ]

    if pd.isna(header_value):
        header_row = 0
    else:
        header_row = int(
            float(header_value)
        )

    loader_type = str(
        row["loader_type"]
    ).strip()

    if loader_type == "CSV":
        X = pd.read_csv(
            source_path,
            sep=separator,
            header=header_row,
        )

        target_value = row[
            "target_column"
        ]

        if pd.isna(target_value):
            raise ValueError(
                "target_column is empty for '{}'.".format(
                    dataset_name
                )
            )

        target_column = str(
            target_value
        ).strip()

        if target_column not in X.columns:
            raise ValueError(
                "Target '{}' not found in '{}'. Columns: {}".format(
                    target_column,
                    dataset_name,
                    list(X.columns),
                )
            )

        y_raw = X[
            target_column
        ].copy()

        X = X.drop(
            columns=[target_column]
        )

    elif loader_type == "CSV_PAIR":
        X = pd.read_csv(
            source_path,
            sep=separator,
            header=header_row,
        )

        target_file_value = row[
            "target_file"
        ]

        if pd.isna(target_file_value):
            raise ValueError(
                "target_file is empty for '{}'.".format(
                    dataset_name
                )
            )

        target_file = str(
            target_file_value
        ).strip()

        target_path = RAW_DIR / target_file

        if not target_path.exists():
            raise FileNotFoundError(
                "Target file not found: {}".format(
                    target_path
                )
            )

        target_df = pd.read_csv(
            target_path
        )

        if target_df.shape[1] != 1:
            raise ValueError(
                "Expected one target column in '{}', "
                "found {}.".format(
                    target_file,
                    target_df.shape[1],
                )
            )

        y_raw = target_df.iloc[
            :,
            0,
        ]

    elif loader_type == "CSV_HEADERLESS":
        X_raw = pd.read_csv(
            source_path,
            sep=separator,
            header=None,
        )

        target_index_value = row[
            "target_column_index"
        ]

        if pd.isna(
            target_index_value
        ):
            raise ValueError(
                "target_column_index is empty for '{}'.".format(
                    dataset_name
                )
            )

        target_index = int(
            float(
                target_index_value
            )
        )

        if (
            target_index < 0
            or target_index
            >= X_raw.shape[1]
        ):
            raise ValueError(
                "Target index {} outside {} columns.".format(
                    target_index,
                    X_raw.shape[1],
                )
            )

        y_raw = X_raw.iloc[
            :,
            target_index,
        ].copy()

        X = X_raw.drop(
            columns=[target_index]
        ).copy()

        X.columns = [
            "feature_{}".format(
                index
            )
            for index
            in range(
                X.shape[1]
            )
        ]

    else:
        raise ValueError(
            "Unsupported CSV loader_type '{}'.".format(
                loader_type
            )
        )

    if len(X) != len(y_raw):
        raise ValueError(
            "Feature/target length mismatch for '{}': "
            "{} vs {}".format(
                dataset_name,
                len(X),
                len(y_raw),
            )
        )

    y, majority_class, minority_class = (
        parse_target_labels(y_raw)
    )

    numeric_features = []
    categorical_features = []

    for column in X.columns:
        numeric = pd.api.types.is_numeric_dtype(
            X[column]
        )

        if numeric:
            numeric_features.append(
                column
            )
        else:
            categorical_features.append(
                column
            )

    for column in numeric_features:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    for column in categorical_features:
        X[column] = (
            X[column]
            .astype(str)
            .str.strip()
        )

    if X.isnull().any().any():
        missing = X.columns[
            X.isnull().any()
        ].tolist()

        raise ValueError(
            "Missing/invalid feature values in '{}': {}".format(
                dataset_name,
                missing,
            )
        )

    return (
        X,
        y,
        majority_class,
        minority_class,
        numeric_features,
        categorical_features,
    )


def load_registry_dataset(
    dataset_name,
):
    registry = read_registry()

    row = get_registry_row(
        registry,
        dataset_name,
    )

    source_value = row[
        "source_file"
    ]

    if pd.isna(source_value):
        raise ValueError(
            "Dataset '{}' has no source_file in registry.".format(
                dataset_name
            )
        )

    loader_type = str(
        row["loader_type"]
    ).strip()

    if loader_type in (
        "KEEL_DAT",
        "KEEL_ZIP",
    ):
        return prepare_keel_dataset(
            row,
            dataset_name,
        )

    return prepare_csv_dataset(
        row,
        dataset_name,
    )


def run_dataset(
    dataset_name,
):
    (
        X,
        y,
        majority_class,
        minority_class,
        numeric_features,
        categorical_features,
    ) = load_registry_dataset(
        dataset_name
    )

    print("=" * 75)
    print(
        "ResampleRank - Registry Benchmark"
    )
    print("=" * 75)

    print(
        "\nDataset           : {}".format(
            dataset_name
        )
    )

    print(
        "Samples           : {}".format(
            X.shape[0]
        )
    )

    print(
        "Features          : {}".format(
            X.shape[1]
        )
    )

    print(
        "Numeric features  : {}".format(
            len(numeric_features)
        )
    )

    print(
        "Categorical       : {}".format(
            len(categorical_features)
        )
    )

    print(
        "Majority class    : {} -> 0".format(
            majority_class
        )
    )

    print(
        "Minority class    : {} -> 1".format(
            minority_class
        )
    )

    classifiers = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
    ]

    all_results = []

    for classifier_name in classifiers:
        result_df = benchmark.run_classifier_benchmark(
            dataset_name,
            X,
            y,
            numeric_features,
            categorical_features,
            classifier_name,
        )

        all_results.append(
            result_df
        )

    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    combined_file = (
        benchmark.RESULTS_DIR
        / "{}_all_results.csv".format(
            dataset_name
        )
    )

    combined.to_csv(
        combined_file,
        index=False,
    )

    successful = (
        combined["verification"]
        == "PASS"
    ).sum()

    failures = (
        combined["verification"]
        == "FAIL"
    ).sum()

    not_applicable = (
        combined["verification"]
        == "NOT_APPLICABLE"
    ).sum()

    valid_macro_f1 = combined.loc[
        combined["verification"] == "PASS",
        "macro_f1_math",
    ]

    print()
    print("=" * 75)
    print("REGISTRY BENCHMARK SUMMARY")
    print("=" * 75)

    print(
        "Total experiments      : {}".format(
            len(combined)
        )
    )

    print(
        "Successful experiments : {}".format(
            successful
        )
    )

    print(
        "Verification failures  : {}".format(
            failures
        )
    )

    print(
        "Not applicable         : {}".format(
            not_applicable
        )
    )

    if not valid_macro_f1.empty:
        print(
            "Macro-F1 minimum       : {:.6f}".format(
                valid_macro_f1.min()
            )
        )

        print(
            "Macro-F1 maximum       : {:.6f}".format(
                valid_macro_f1.max()
            )
        )

    print(
        "\nCombined results saved to:"
    )

    print(
        combined_file
    )

    if failures != 0:
        raise RuntimeError(
            "Benchmark produced {} verification failures.".format(
                failures
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a ResampleRank benchmark "
            "using dataset_registry.csv."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name from dataset_registry.csv.",
    )

    args = parser.parse_args()

    run_dataset(
        args.dataset
    )


if __name__ == "__main__":
    main()
