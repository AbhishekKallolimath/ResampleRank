from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"


def convert_keel_dataset(dataset_name):
    input_file = RAW_DIR / f"{dataset_name}.dat"
    feature_file = RAW_DIR / f"{dataset_name}.csv"
    target_file = RAW_DIR / f"{dataset_name}_target.csv"

    if not input_file.exists():
        raise FileNotFoundError(f"Dataset not found: {input_file}")

    data_started = False
    rows = []
    feature_names = []
    feature_types = {}
    target_name = None

    with open(input_file, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            lower_line = line.lower()

            # Read attribute definitions
            if lower_line.startswith("@attribute"):
                parts = line.split(maxsplit=2)

                if len(parts) < 3:
                    raise ValueError(f"Invalid attribute definition: {line}")

                name = parts[1]
                definition = parts[2].strip()

                if name.lower() == "class":
                    target_name = name
                    feature_types[name] = "target"

                else:
                    feature_names.append(name)

                    # KEEL nominal/categorical attribute:
                    # @attribute Sex {M, F, I}
                    # @attribute Legs {0,2,4,5,6,8}
                    if definition.startswith("{") and definition.endswith("}"):
                        feature_types[name] = "categorical"

                    # Numeric attributes:
                    # @attribute Length real [...]
                    # @attribute Age integer [...]
                    elif definition.lower().startswith(
                        ("real", "integer")
                    ):
                        feature_types[name] = "numeric"

                    else:
                        raise ValueError(
                            f"Unsupported attribute type for "
                            f"'{name}': {definition}"
                        )

            elif lower_line == "@data":
                data_started = True

            elif data_started and not line.startswith("%"):
                rows.append(
                    [value.strip() for value in line.split(",")]
                )

    if not rows:
        raise ValueError("No data rows found.")

    if target_name is None:
        raise ValueError("Target attribute not found.")

    expected_columns = len(feature_names) + 1

    # Validate row lengths
    for index, row in enumerate(rows, start=1):
        if len(row) != expected_columns:
            raise ValueError(
                f"Row {index} has {len(row)} values; "
                f"expected {expected_columns}."
            )

    feature_rows = [row[:-1] for row in rows]
    target_rows = [row[-1] for row in rows]

    features = pd.DataFrame(
        feature_rows,
        columns=feature_names
    )

    targets = pd.DataFrame(
        target_rows,
        columns=["class"]
    )

    # Convert features according to the original KEEL attribute type
    for column in features.columns:
        column_type = feature_types[column]

        if column_type == "numeric":
            original_values = features[column].copy()

            converted = pd.to_numeric(
                original_values,
                errors="coerce"
            )

            # Detect invalid numeric values
            invalid_mask = (
                converted.isna()
                & original_values.notna()
                & (original_values.astype(str).str.strip() != "")
            )

            if invalid_mask.any():
                invalid_values = original_values[invalid_mask].unique()

                raise ValueError(
                    f"Column '{column}' contains invalid numeric "
                    f"values: {invalid_values}"
                )

            features[column] = converted

        elif column_type == "categorical":
            features[column] = (
                features[column]
                .astype("string")
                .str.strip()
            )

    # Detect missing feature values
    if features.isnull().any().any():
        missing_columns = features.columns[
            features.isnull().any()
        ].tolist()

        raise ValueError(
            f"Feature data contains missing values in: "
            f"{missing_columns}"
        )

    # Detect missing targets
    targets["class"] = (
        targets["class"]
        .astype("string")
        .str.strip()
    )

    if targets["class"].isnull().any():
        raise ValueError("Target contains missing values.")

    # Save files
    features.to_csv(feature_file, index=False)
    targets.to_csv(target_file, index=False)

    numeric_columns = [
        column
        for column in feature_names
        if feature_types[column] == "numeric"
    ]

    categorical_columns = [
        column
        for column in feature_names
        if feature_types[column] == "categorical"
    ]

    print("=" * 70)
    print(f"Converted dataset: {dataset_name}")
    print("=" * 70)
    print(f"Samples          : {len(features)}")
    print(f"Features         : {len(features.columns)}")
    print(f"Numeric features : {len(numeric_columns)}")
    print(f"Categorical      : {len(categorical_columns)}")
    print(f"Target           : {target_name}")
    print(f"Classes          : {sorted(targets['class'].unique())}")
    print(f"Feature file     : {feature_file}")
    print(f"Target file      : {target_file}")

    if categorical_columns:
        print(f"Categorical cols : {categorical_columns}")

    print("=" * 70)


if __name__ == "__main__":
    convert_keel_dataset("zoo-3")