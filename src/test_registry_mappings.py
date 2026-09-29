
from pathlib import Path
import zipfile

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
REGISTRY_FILE = BASE_DIR / "data" / "dataset_registry.csv"


def load_registry():
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(
            "Registry not found: {}".format(
                REGISTRY_FILE
            )
        )

    registry = pd.read_csv(
        REGISTRY_FILE
    )

    required_columns = [
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
        for column in required_columns
        if column not in registry.columns
    ]

    if missing:
        raise ValueError(
            "Missing registry columns: {}".format(
                missing
            )
        )

    return registry


def verify_keel_content(
    content,
    dataset,
    expected_target,
):
    text = content.decode(
        "utf-8",
        errors="replace"
    )

    lines = text.splitlines()

    attribute_names = []

    for line in lines:
        stripped = line.strip()

        if not stripped.lower().startswith(
            "@attribute"
        ):
            continue

        parts = stripped.split(
            maxsplit=2
        )

        if len(parts) >= 2:
            attribute_names.append(
                parts[1].strip()
            )

    if not attribute_names:
        raise ValueError(
            "No @attribute definitions found"
        )

    data_marker_found = any(
        line.strip().lower() == "@data"
        for line in lines
    )

    if not data_marker_found:
        raise ValueError(
            "KEEL @data marker not found"
        )

    expected_target = str(
        expected_target
    ).strip()

    if expected_target:
        target_match = any(
            name.lower()
            == expected_target.lower()
            for name in attribute_names
        )

        if not target_match:
            raise ValueError(
                "KEEL target '{}' not found. "
                "Attributes: {}".format(
                    expected_target,
                    attribute_names,
                )
            )

    return (
        "KEEL dataset '{}' verified; "
        "{} attributes found; "
        "target='{}'"
    ).format(
        dataset,
        len(attribute_names),
        expected_target,
    )


def test_dataset(row):
    dataset = str(
        row["dataset"]
    ).strip()

    source_value = row["source_file"]

    if pd.isna(source_value):
        return {
            "dataset": dataset,
            "status": "SKIPPED_NO_SOURCE",
            "detail": "source_file is empty",
        }

    source_file = str(
        source_value
    ).strip()

    if (
        not source_file
        or source_file.lower() == "nan"
    ):
        return {
            "dataset": dataset,
            "status": "SKIPPED_NO_SOURCE",
            "detail": "source_file is empty",
        }

    source_path = RAW_DIR / source_file

    if not source_path.exists():
        return {
            "dataset": dataset,
            "status": "FAIL",
            "detail": "File not found: {}".format(
                source_path
            ),
        }

    loader_value = row["loader_type"]

    if pd.isna(loader_value):
        return {
            "dataset": dataset,
            "status": "FAIL",
            "detail": "loader_type is empty",
        }

    loader_type = str(
        loader_value
    ).strip()

    separator_value = row["separator"]

    if pd.isna(separator_value):
        separator = ","
    else:
        separator = str(
            separator_value
        )

    try:

        if loader_type == "CSV":
            header_value = row[
                "header_row"
            ]

            if pd.isna(header_value):
                header_row = 0
            else:
                header_row = int(
                    float(header_value)
                )

            sample = pd.read_csv(
                source_path,
                sep=separator,
                header=header_row,
                nrows=5,
            )

            target_value = row[
                "target_column"
            ]

            if pd.isna(target_value):
                raise ValueError(
                    "target_column is empty"
                )

            target_column = str(
                target_value
            ).strip()

            if target_column not in (
                sample.columns
            ):
                raise ValueError(
                    "Target column '{}' not found. "
                    "Columns: {}".format(
                        target_column,
                        list(sample.columns),
                    )
                )

            return {
                "dataset": dataset,
                "status": "PASS",
                "detail": (
                    "CSV target '{}' verified"
                ).format(
                    target_column
                ),
            }

        if loader_type == "CSV_PAIR":
            header_value = row[
                "header_row"
            ]

            if pd.isna(header_value):
                header_row = 0
            else:
                header_row = int(
                    float(header_value)
                )

            feature_sample = pd.read_csv(
                source_path,
                sep=separator,
                header=header_row,
                nrows=5,
            )

            target_file_value = row[
                "target_file"
            ]

            if pd.isna(target_file_value):
                raise ValueError(
                    "target_file is empty"
                )

            target_file = str(
                target_file_value
            ).strip()

            if (
                not target_file
                or target_file.lower()
                == "nan"
            ):
                raise ValueError(
                    "target_file is empty"
                )

            target_path = (
                RAW_DIR / target_file
            )

            if not target_path.exists():
                raise ValueError(
                    "Target file not found: {}".format(
                        target_path
                    )
                )

            target_sample = pd.read_csv(
                target_path,
                nrows=5,
            )

            if target_sample.shape[1] != 1:
                raise ValueError(
                    "Expected one target column, "
                    "found {}".format(
                        target_sample.shape[1]
                    )
                )

            target_value = row[
                "target_column"
            ]

            if pd.isna(target_value):
                raise ValueError(
                    "target_column is empty"
                )

            target_column = str(
                target_value
            ).strip()

            if target_column not in (
                target_sample.columns
            ):
                raise ValueError(
                    "Target column '{}' not found "
                    "in target file. Columns: {}".format(
                        target_column,
                        list(
                            target_sample.columns
                        ),
                    )
                )

            return {
                "dataset": dataset,
                "status": "PASS",
                "detail": (
                    "CSV_PAIR verified; "
                    "features={} columns, "
                    "target='{}', "
                    "target file='{}'"
                ).format(
                    feature_sample.shape[1],
                    target_column,
                    target_file,
                ),
            }

        if loader_type == "CSV_HEADERLESS":
            target_index_value = row[
                "target_column_index"
            ]

            if pd.isna(
                target_index_value
            ):
                raise ValueError(
                    "target_column_index is empty"
                )

            target_index = int(
                float(target_index_value)
            )

            sample = pd.read_csv(
                source_path,
                sep=separator,
                header=None,
                nrows=5,
            )

            if (
                target_index < 0
                or target_index
                >= sample.shape[1]
            ):
                raise ValueError(
                    "Target index {} outside "
                    "{} columns".format(
                        target_index,
                        sample.shape[1],
                    )
                )

            target_values = (
                sample.iloc[
                    :,
                    target_index
                ]
            )

            return {
                "dataset": dataset,
                "status": "PASS",
                "detail": (
                    "Headerless CSV target "
                    "index {} verified; "
                    "sample target values={}"
                ).format(
                    target_index,
                    target_values.head(
                        5
                    ).tolist(),
                ),
            }

        if loader_type == "KEEL_DAT":
            with open(
                source_path,
                "rb"
            ) as file:
                content = file.read()

            target_value = row[
                "target_column"
            ]

            if pd.isna(target_value):
                expected_target = ""
            else:
                expected_target = str(
                    target_value
                ).strip()

            detail = verify_keel_content(
                content,
                dataset,
                expected_target,
            )

            return {
                "dataset": dataset,
                "status": "PASS",
                "detail": detail,
            }

        if loader_type == "KEEL_ZIP":
            with zipfile.ZipFile(
                source_path,
                "r"
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
                        "No .dat file found inside ZIP"
                    )

                dat_name = entries[0]

                with archive.open(
                    dat_name,
                    "r"
                ) as file:
                    content = file.read()

            target_value = row[
                "target_column"
            ]

            if pd.isna(target_value):
                expected_target = ""
            else:
                expected_target = str(
                    target_value
                ).strip()

            detail = verify_keel_content(
                content,
                dataset,
                expected_target,
            )

            return {
                "dataset": dataset,
                "status": "PASS",
                "detail": (
                    "{}; archive='{}'"
                ).format(
                    detail,
                    dat_name,
                ),
            }

        raise ValueError(
            "Unsupported loader_type '{}'".format(
                loader_type
            )
        )

    except Exception as exc:
        return {
            "dataset": dataset,
            "status": "FAIL",
            "detail": str(exc),
        }


def main():
    registry = load_registry()

    results = []

    for _, row in registry.iterrows():
        results.append(
            test_dataset(row)
        )

    results_df = pd.DataFrame(
        results
    )

    print("=" * 80)
    print(
        "RESAMPLERANK REGISTRY MAPPING TEST"
    )
    print("=" * 80)

    datasets_with_source = (
        registry[
            "source_file"
        ].notna().sum()
    )

    print(
        "Datasets with source_file : {}".format(
            datasets_with_source
        )
    )

    print()

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        "PASS            : {}".format(
            (
                results_df["status"]
                == "PASS"
            ).sum()
        )
    )

    print(
        "FAIL            : {}".format(
            (
                results_df["status"]
                == "FAIL"
            ).sum()
        )
    )

    print(
        "SKIPPED         : {}".format(
            (
                results_df["status"]
                == "SKIPPED_NO_SOURCE"
            ).sum()
        )
    )

    failures = results_df[
        results_df["status"] == "FAIL"
    ]

    if not failures.empty:
        print()
        print("FAILURES:")
        print(
            failures.to_string(
                index=False
            )
        )
        raise SystemExit(1)

    print()
    print(
        "Registry mapping test completed successfully."
    )


if __name__ == "__main__":
    main()
