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
    target_name = None

    with open(input_file, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            lower_line = line.lower()

            if lower_line.startswith("@attribute"):
                parts = line.split()
                name = parts[1]

                if name.lower() == "class":
                    target_name = name
                else:
                    feature_names.append(name)

            elif lower_line == "@data":
                data_started = True

            elif data_started and not line.startswith("%"):
                rows.append([value.strip() for value in line.split(",")])

    if not rows:
        raise ValueError("No data rows found.")

    if target_name is None:
        raise ValueError("Target attribute not found.")

    expected_columns = len(feature_names) + 1

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

    for column in features.columns:
        features[column] = pd.to_numeric(features[column])

    if features.isnull().any().any():
        raise ValueError("Feature data contains invalid or missing values.")

    if targets["class"].isnull().any():
        raise ValueError("Target contains missing values.")

    features.to_csv(feature_file, index=False)
    targets.to_csv(target_file, index=False)

    print("=" * 70)
    print(f"Converted dataset: {dataset_name}")
    print("=" * 70)
    print(f"Samples          : {len(features)}")
    print(f"Features         : {len(features.columns)}")
    print(f"Target           : {target_name}")
    print(f"Classes          : {sorted(targets['class'].unique())}")
    print(f"Feature file     : {feature_file}")
    print(f"Target file      : {target_file}")
    print("=" * 70)


if __name__ == "__main__":
    convert_keel_dataset("glass1")