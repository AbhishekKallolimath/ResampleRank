
from pathlib import Path
import csv
import re
import zipfile

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
REGISTRY_FILE = BASE_DIR / "data" / "dataset_registry.csv"
SPLIT_FILE = BASE_DIR / "data" / "dataset_splits.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "dataset_file_audit.csv"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


VALID_SPLITS = {
    "TRAIN",
    "VALIDATION",
    "TEST",
}


PREFERRED_TARGET_NAMES = {
    "class",
    "target",
    "label",
    "outcome",
    "response",
    "result",
    "y",
    "churn",
    "fraud",
    "default",
    "income",
    "diagnosis",
    "status",
    "attrition",
    "stroke",
    "stabf",
    "potability",
    "quality",
}


KNOWN_TARGETS = {
    "pima": "class",
    "glass1": "class",
    "wisconsin": "class",
    "yeast1": "class",
    "zoo-3": "class",
    "abalone9-18": "class",
    "haberman": "class",
    "iris0": "class",
    "segment0": "class",
    "vowel0": "class",
    "vehicle2": "class",
    "page-blocks0": "class",
    "shuttle-c0-vs-c4": "class",
    "ecoli-0_vs_1": "class",
    "glass0": "Class",
    "glass2": "Class",
    "glass4": "Class",
    "glass5": "Class",
    "glass6": "Class",
    "vehicle0": "Class",
    "vehicle1": "Class",
    "vehicle3": "Class",
    "ecoli1": "Class",
    "ecoli2": "Class",
    "ecoli3": "Class",
    "ecoli4": "Class",
    "new-thyroid1": "Class",
    "new-thyroid2": "Class",
}


def normalize_name(value):
    value = str(value).lower().strip()

    value = re.sub(
        r"[^a-z0-9]+",
        "",
        value,
    )

    return value


def load_registry():
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(
            f"Registry not found:\n{REGISTRY_FILE}"
        )

    registry = pd.read_csv(
        REGISTRY_FILE
    )

    required_columns = {
        "dataset",
        "split",
    }

    missing = (
        required_columns
        - set(registry.columns)
    )

    if missing:
        raise ValueError(
            "dataset_registry.csv is missing columns: "
            f"{sorted(missing)}"
        )

    registry["dataset"] = (
        registry["dataset"]
        .astype(str)
        .str.strip()
    )

    registry["split"] = (
        registry["split"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    invalid = (
        set(registry["split"])
        - VALID_SPLITS
    )

    if invalid:
        raise ValueError(
            f"Invalid split values in registry: "
            f"{sorted(invalid)}"
        )

    return registry


def load_splits():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset split file not found:\n"
            f"{SPLIT_FILE}"
        )

    splits = pd.read_csv(
        SPLIT_FILE
    )

    required_columns = {
        "dataset",
        "split",
    }

    missing = (
        required_columns
        - set(splits.columns)
    )

    if missing:
        raise ValueError(
            "dataset_splits.csv is missing columns: "
            f"{sorted(missing)}"
        )

    splits["dataset"] = (
        splits["dataset"]
        .astype(str)
        .str.strip()
    )

    splits["split"] = (
        splits["split"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if splits["dataset"].duplicated().any():
        duplicates = sorted(
            splits.loc[
                splits["dataset"].duplicated(
                    keep=False
                ),
                "dataset",
            ].unique()
        )

        raise ValueError(
            "Duplicate dataset entries in "
            "dataset_splits.csv:\n"
            + "\n".join(duplicates)
        )

    return splits


def collect_raw_files():
    files = []

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw directory not found:\n{RAW_DIR}"
        )

    for path in RAW_DIR.rglob("*"):

        if not path.is_file():
            continue

        relative_parts = [
            part.lower()
            for part in path.relative_to(
                RAW_DIR
            ).parts
        ]

        if "__macosx" in relative_parts:
            continue

        if path.name.startswith("._"):
            continue

        suffix = path.suffix.lower()

        if suffix not in {
            ".csv",
            ".dat",
            ".zip",
        }:
            continue

        files.append(path)

    return sorted(
        files,
        key=lambda p: str(p).lower()
    )


def find_standard_pair(
    dataset_name,
):
    feature_file = (
        RAW_DIR
        / f"{dataset_name}.csv"
    )

    target_file = (
        RAW_DIR
        / f"{dataset_name}_target.csv"
    )

    if (
        feature_file.exists()
        and target_file.exists()
    ):
        return (
            feature_file,
            target_file,
        )

    return (
        None,
        None,
    )


def parse_keel_attribute_line(line):
    parts = line.strip().split(
        maxsplit=2
    )

    if len(parts) < 3:
        return None, None

    name = parts[1].strip()
    definition = parts[2].strip()

    if (
        name.startswith("'")
        and name.endswith("'")
    ):
        name = name[1:-1]

    if (
        name.startswith('"')
        and name.endswith('"')
    ):
        name = name[1:-1]

    return (
        name,
        definition,
    )


def parse_keel_file(
    file_path,
):
    attributes = []
    attribute_types = {}

    target_column = None

    data_started = False
    row_count = 0

    class_values = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:

        for raw_line in file:

            line = raw_line.strip()

            if not line:
                continue

            lower = line.lower()

            if lower == "@data":
                data_started = True
                continue

            if lower.startswith(
                "@attribute"
            ):

                name, definition = (
                    parse_keel_attribute_line(
                        line
                    )
                )

                if name is None:
                    continue

                attributes.append(
                    name
                )

                if (
                    definition.startswith("{")
                    and
                    definition.endswith("}")
                ):
                    attribute_types[name] = (
                        "categorical"
                    )
                else:
                    attribute_types[name] = (
                        "numeric"
                    )

                if name.lower() in {
                    "class",
                    "target",
                    "label",
                    "outcome",
                }:
                    target_column = name

                continue

            if (
                data_started
                and not line.startswith("%")
            ):

                row_count += 1

                if (
                    target_column is not None
                    and len(
                        class_values
                    ) < 5000
                ):

                    reader = csv.reader(
                        [line]
                    )

                    values = next(
                        reader
                    )

                    values = [
                        value.strip()
                        for value in values
                    ]

                    if len(values) == len(
                        attributes
                    ):

                        target_index = (
                            attributes.index(
                                target_column
                            )
                        )

                        class_values.append(
                            values[
                                target_index
                            ]
                        )

    if target_column is None:
        target_column = (
            attributes[-1]
            if attributes
            else ""
        )

    numeric_features = [
        column
        for column in attributes
        if (
            column != target_column
            and
            attribute_types.get(column)
            == "numeric"
        )
    ]

    categorical_features = [
        column
        for column in attributes
        if (
            column != target_column
            and
            attribute_types.get(column)
            == "categorical"
        )
    ]

    unique_classes = sorted(
        set(
            value
            for value in class_values
            if value != ""
        )
    )

    binary = (
        len(unique_classes) == 2
    )

    return {
        "format": "KEEL_DAT",
        "file": str(
            file_path.relative_to(
                BASE_DIR
            )
        ),
        "target_column_detected": (
            target_column
        ),
        "samples": row_count,
        "features": max(
            0,
            len(attributes) - 1,
        ),
        "numeric_features": len(
            numeric_features
        ),
        "categorical_features": len(
            categorical_features
        ),
        "classes_in_sample": len(
            unique_classes
        ),
        "binary_in_sample": binary,
        "target_candidates": (
            target_column
        ),
        "status": (
            "READY"
            if (
                target_column
                and binary
                and row_count > 0
            )
            else "REVIEW"
        ),
        "notes": (
            "Loaded directly from KEEL .dat"
        ),
    }


def get_csv_target_candidates(
    file_path,
    known_target=None,
):
    try:
        df = pd.read_csv(
            file_path,
            nrows=5000,
        )
    except Exception as error:
        return {
            "format": "CSV",
            "file": str(
                file_path.relative_to(
                    BASE_DIR
                )
            ),
            "target_column_detected": "",
            "samples": "",
            "features": "",
            "numeric_features": "",
            "categorical_features": "",
            "classes_in_sample": "",
            "binary_in_sample": "",
            "target_candidates": "",
            "status": "READ_ERROR",
            "notes": str(error),
        }

    target_column = ""

    if known_target in df.columns:
        target_column = known_target

    numeric_features = [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(
            df[column]
        )
    ]

    categorical_features = [
        column
        for column in df.columns
        if column not in numeric_features
    ]

    binary_candidates = []

    for column in df.columns:

        series = (
            df[column]
            .dropna()
            .astype(str)
            .str.strip()
        )

        unique_count = (
            series.nunique()
        )

        if unique_count == 2:
            binary_candidates.append(
                column
            )

    if not target_column:

        preferred_candidates = [
            column
            for column in binary_candidates
            if column.lower()
            in PREFERRED_TARGET_NAMES
        ]

        if len(
            preferred_candidates
        ) == 1:

            target_column = (
                preferred_candidates[0]
            )

        elif len(
            binary_candidates
        ) == 1:

            target_column = (
                binary_candidates[0]
            )

    classes_in_sample = ""

    if target_column:
        classes_in_sample = (
            df[target_column]
            .dropna()
            .astype(str)
            .str.strip()
            .nunique()
        )

    binary = (
        classes_in_sample == 2
        if classes_in_sample != ""
        else ""
    )

    if target_column and binary:
        status = "READY"
    elif binary_candidates:
        status = "NEEDS_TARGET_SELECTION"
    else:
        status = "REVIEW"

    notes = (
        "Target detected using registry/"
        "target-name rules"
    )

    return {
        "format": "CSV",
        "file": str(
            file_path.relative_to(
                BASE_DIR
            )
        ),
        "target_column_detected": (
            target_column
        ),
        "samples": "",
        "features": len(
            df.columns
        ),
        "numeric_features": len(
            numeric_features
        ),
        "categorical_features": len(
            categorical_features
        ),
        "classes_in_sample": (
            classes_in_sample
        ),
        "binary_in_sample": binary,
        "target_candidates": (
            ", ".join(
                binary_candidates
            )
        ),
        "status": status,
        "notes": notes,
    }


def inspect_zip(
    file_path,
):
    try:
        with zipfile.ZipFile(
            file_path,
            "r",
        ) as archive:

            members = archive.namelist()

            clean_members = [
                member
                for member in members
                if "__MACOSX/"
                not in member
                and
                not Path(member).name.startswith(
                    "._"
                )
            ]

            data_members = [
                member
                for member in clean_members
                if member.lower().endswith(
                    (
                        ".csv",
                        ".dat",
                        ".data",
                        ".txt",
                    )
                )
            ]

            if data_members:

                return {
                    "format": "ZIP",
                    "file": str(
                        file_path.relative_to(
                            BASE_DIR
                        )
                    ),
                    "target_column_detected": "",
                    "samples": "",
                    "features": "",
                    "numeric_features": "",
                    "categorical_features": "",
                    "classes_in_sample": "",
                    "binary_in_sample": "",
                    "target_candidates": "",
                    "status": "ZIP_ONLY",
                    "notes": (
                        "Archive contains data files: "
                        + "; ".join(
                            data_members[:10]
                        )
                    ),
                }

            return {
                "format": "ZIP",
                "file": str(
                    file_path.relative_to(
                        BASE_DIR
                    )
                ),
                "target_column_detected": "",
                "samples": "",
                "features": "",
                "numeric_features": "",
                "categorical_features": "",
                "classes_in_sample": "",
                "binary_in_sample": "",
                "target_candidates": "",
                "status": "ZIP_REVIEW",
                "notes": (
                    "No CSV/DAT data file detected"
                ),
            }

    except zipfile.BadZipFile:

        return {
            "format": "ZIP",
            "file": str(
                file_path.relative_to(
                    BASE_DIR
                )
            ),
            "target_column_detected": "",
            "samples": "",
            "features": "",
            "numeric_features": "",
            "categorical_features": "",
            "classes_in_sample": "",
            "binary_in_sample": "",
            "target_candidates": "",
            "status": "ZIP_READ_ERROR",
            "notes": "Invalid ZIP archive",
        }

    except Exception as error:

        return {
            "format": "ZIP",
            "file": str(
                file_path.relative_to(
                    BASE_DIR
                )
            ),
            "target_column_detected": "",
            "samples": "",
            "features": "",
            "numeric_features": "",
            "categorical_features": "",
            "classes_in_sample": "",
            "binary_in_sample": "",
            "target_candidates": "",
            "status": "ZIP_READ_ERROR",
            "notes": str(error),
        }


def find_keel_files(
    dataset_name,
    raw_files,
):
    normalized_dataset = (
        normalize_name(
            dataset_name
        )
    )

    matches = []

    for path in raw_files:

        if path.suffix.lower() != ".dat":
            continue

        stem_normalized = (
            normalize_name(
                path.stem
            )
        )

        if (
            stem_normalized
            == normalized_dataset
        ):
            matches.append(
                path
            )

    return matches


def find_candidate_files(
    dataset_name,
    raw_files,
):
    normalized_dataset = (
        normalize_name(
            dataset_name
        )
    )

    matches = []

    for path in raw_files:

        normalized_name = (
            normalize_name(
                path.name
            )
        )

        normalized_stem = (
            normalize_name(
                path.stem
            )
        )

        if (
            normalized_name
            == normalized_dataset
            or
            normalized_stem
            == normalized_dataset
        ):
            matches.append(
                path
            )

    return matches


def inspect_dataset(
    dataset_name,
    raw_files,
):
    known_target = (
        KNOWN_TARGETS.get(
            dataset_name
        )
    )

    feature_file, target_file = (
        find_standard_pair(
            dataset_name
        )
    )

    if (
        feature_file is not None
        and target_file is not None
    ):

        try:

            X = pd.read_csv(
                feature_file,
                nrows=5000,
            )

            y = pd.read_csv(
                target_file,
                nrows=5000,
            )

            if y.shape[1] != 1:

                return {
                    "format": "STANDARD_PAIR",
                    "file": str(
                        feature_file.relative_to(
                            BASE_DIR
                        )
                    ),
                    "target_column_detected": "",
                    "samples": "",
                    "features": len(
                        X.columns
                    ),
                    "numeric_features": "",
                    "categorical_features": "",
                    "classes_in_sample": "",
                    "binary_in_sample": "",
                    "target_candidates": "",
                    "status": "REVIEW",
                    "notes": (
                        "Target file does not "
                        "contain exactly one column"
                    ),
                }

            target_column = (
                y.columns[0]
            )

            target_series = (
                y.iloc[:, 0]
                .dropna()
                .astype(str)
                .str.strip()
            )

            class_count = (
                target_series.nunique()
            )

            return {
                "format": "STANDARD_PAIR",
                "file": str(
                    feature_file.relative_to(
                        BASE_DIR
                    )
                ),
                "target_column_detected": (
                    target_column
                ),
                "samples": (
                    len(y)
                ),
                "features": len(
                    X.columns
                ),
                "numeric_features": len(
                    X.select_dtypes(
                        include="number"
                    ).columns
                ),
                "categorical_features": (
                    len(X.columns)
                    -
                    len(
                        X.select_dtypes(
                            include="number"
                        ).columns
                    )
                ),
                "classes_in_sample": (
                    class_count
                ),
                "binary_in_sample": (
                    class_count == 2
                ),
                "target_candidates": (
                    target_column
                ),
                "status": (
                    "READY"
                    if class_count == 2
                    else "REVIEW"
                ),
                "notes": (
                    "Existing feature + target pair"
                ),
            }

        except Exception as error:

            return {
                "format": "STANDARD_PAIR",
                "file": str(
                    feature_file.relative_to(
                        BASE_DIR
                    )
                ),
                "target_column_detected": "",
                "samples": "",
                "features": "",
                "numeric_features": "",
                "categorical_features": "",
                "classes_in_sample": "",
                "binary_in_sample": "",
                "target_candidates": "",
                "status": "READ_ERROR",
                "notes": str(error),
            }

    keel_files = find_keel_files(
        dataset_name,
        raw_files,
    )

    if keel_files:

        return parse_keel_file(
            keel_files[0]
        )

    candidates = find_candidate_files(
        dataset_name,
        raw_files,
    )

    if candidates:

        csv_files = [
            path
            for path in candidates
            if path.suffix.lower()
            == ".csv"
        ]

        zip_files = [
            path
            for path in candidates
            if path.suffix.lower()
            == ".zip"
        ]

        if csv_files:

            preferred_csv = (
                csv_files[0]
            )

            return get_csv_target_candidates(
                preferred_csv,
                known_target=known_target,
            )

        if zip_files:

            return inspect_zip(
                zip_files[0]
            )

    return {
        "format": "",
        "file": "",
        "target_column_detected": "",
        "samples": "",
        "features": "",
        "numeric_features": "",
        "categorical_features": "",
        "classes_in_sample": "",
        "binary_in_sample": "",
        "target_candidates": "",
        "status": "NOT_FOUND",
        "notes": (
            "No matching usable CSV/DAT/ZIP found"
        ),
    }


def main():

    print("=" * 75)
    print(
        "ResampleRank - Dataset File Audit"
    )
    print("=" * 75)

    registry = load_registry()
    splits = load_splits()

    split_map = (
        splits
        .drop_duplicates(
            subset=["dataset"]
        )
        .set_index("dataset")[
            "split"
        ]
        .to_dict()
    )

    registry["split"] = (
        registry["dataset"]
        .map(split_map)
    )

    if registry["split"].isna().any():

        missing = sorted(
            registry.loc[
                registry["split"].isna(),
                "dataset",
            ].unique()
        )

        raise ValueError(
            "Datasets missing from "
            "dataset_splits.csv:\n"
            + "\n".join(missing)
        )

    raw_files = collect_raw_files()

    print(
        f"\nDatasets in registry : "
        f"{len(registry)}"
    )

    print(
        f"Usable files discovered : "
        f"{len(raw_files)}"
    )

    rows = []

    for index, row in registry.iterrows():

        dataset_name = (
            row["dataset"]
        )

        print(
            f"\n[{index + 1:02d}/"
            f"{len(registry):02d}] "
            f"{dataset_name}"
        )

        inspection = inspect_dataset(
            dataset_name,
            raw_files,
        )

        result = {
            "dataset": dataset_name,
            "split": row["split"],
            "repository": row.get(
                "repository",
                "",
            ),
            "dataset_id": row.get(
                "dataset_id",
                "",
            ),
            **inspection,
        }

        rows.append(
            result
        )

        print(
            f"  Status : "
            f"{inspection['status']}"
        )

        if inspection["file"]:

            print(
                f"  File   : "
                f"{inspection['file']}"
            )

        if inspection[
            "target_column_detected"
        ]:

            print(
                f"  Target : "
                f"{inspection['target_column_detected']}"
            )

        if inspection[
            "target_candidates"
        ]:

            print(
                f"  Target candidates: "
                f"{inspection['target_candidates']}"
            )

    audit = pd.DataFrame(
        rows
    )

    audit = (
        audit
        .sort_values(
            [
                "split",
                "dataset",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    audit.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n")
    print("=" * 75)
    print("AUDIT SUMMARY")
    print("=" * 75)

    print(
        f"Total datasets : "
        f"{len(audit)}"
    )

    print(
        "\nStatus counts:"
    )

    print(
        audit["status"]
        .value_counts()
        .to_string()
    )

    print(
        "\nStatus by split:"
    )

    status_by_split = (
        pd.crosstab(
            audit["split"],
            audit["status"],
        )
        .reindex(
            ["TRAIN", "VALIDATION", "TEST"],
            fill_value=0,
        )
    )

    print(
        status_by_split.to_string()
    )

    print(
        "\nDatasets requiring review:"
    )

    review = audit[
        audit["status"]
        != "READY"
    ]

    if review.empty:

        print(
            "None"
        )

    else:

        print(
            review[
                [
                    "dataset",
                    "split",
                    "format",
                    "file",
                    "target_column_detected",
                    "target_candidates",
                    "status",
                    "notes",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\nSaved audit to:"
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()

