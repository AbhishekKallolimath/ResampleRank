
from pathlib import Path
import shutil

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
REGISTRY_FILE = BASE_DIR / "data" / "dataset_registry.csv"
BACKUP_FILE = BASE_DIR / "data" / "dataset_registry.csv.before_keel_mappings.bak"


MAPPINGS = {
    "haberman": {
        "source_file": "haberman.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },
    "iris0": {
        "source_file": "iris0.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },
    "pima": {
        "source_file": "pima.csv",
        "target_file": "pima_target.csv",
        "loader_type": "CSV_PAIR",
        "separator": ",",
        "header_row": 0,
        "target_column": "Outcome",
    },
    "segment0": {
        "source_file": "segment0.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },
    "vowel0": {
        "source_file": "vowel0.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },
    "wisconsin": {
        "source_file": "wisconsin.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "Class",
    },
    "zoo-3": {
        "source_file": "zoo-3.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },

    "glass-0-1-5_vs_2": {
        "source_file": "imb_IRhigherThan9p2/glass-0-1-5_vs_2/glass-0-1-5_vs_2.zip",
        "loader_type": "KEEL_ZIP",
        "target_column": "class",
    },
    "glass-0-1-6_vs_2": {
        "source_file": "imb_IRhigherThan9p1/glass-0-1-6_vs_2/glass-0-1-6_vs_2.zip",
        "loader_type": "KEEL_ZIP",
        "target_column": "class",
    },
    "glass-0-1-6_vs_5": {
        "source_file": "imb_IRhigherThan9p1/glass-0-1-6_vs_5/glass-0-1-6_vs_5.zip",
        "loader_type": "KEEL_ZIP",
        "target_column": "class",
    },
    "glass0": {
        "source_file": "glass0.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },
    "glass1": {
        "source_file": "glass1.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },
    "glass2": {
        "source_file": "imb_IRhigherThan9p1/glass2/glass2.zip",
        "loader_type": "KEEL_ZIP",
        "target_column": "class",
    },
    "glass4": {
        "source_file": "imb_IRhigherThan9p1/glass4/glass4.zip",
        "loader_type": "KEEL_ZIP",
        "target_column": "class",
    },
    "glass5": {
        "source_file": "imb_IRhigherThan9p1/glass5/glass5.zip",
        "loader_type": "KEEL_ZIP",
        "target_column": "class",
    },
    "glass6": {
        "source_file": "glass6.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },

    "vehicle0": {
        "source_file": "vehicle0.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },
    "vehicle1": {
        "source_file": "vehicle1.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },
    "vehicle2": {
        "source_file": "vehicle2.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },
    "vehicle3": {
        "source_file": "vehicle3.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },

    "yeast-0-5-6-7-9_vs_4": {
        "source_file": "imb_IRhigherThan9p1/yeast-0-5-6-7-9_vs_4/yeast-0-5-6-7-9_vs_4.zip",
        "loader_type": "KEEL_ZIP",
        "target_column": "class",
    },
    "yeast-1-2-8-9_vs_7": {
        "source_file": "imb_IRhigherThan9p1/yeast-1-2-8-9_vs_7/yeast-1-2-8-9_vs_7.zip",
        "loader_type": "KEEL_ZIP",
        "target_column": "class",
    },
    "yeast-1-4-5-8_vs_7": {
        "source_file": "imb_IRhigherThan9p1/yeast-1-4-5-8_vs_7/yeast-1-4-5-8_vs_7.zip",
        "loader_type": "KEEL_ZIP",
        "target_column": "class",
    },
    "yeast-1_vs_7": {
        "source_file": "imb_IRhigherThan9p1/yeast-1_vs_7/yeast-1_vs_7.zip",
        "loader_type": "KEEL_ZIP",
        "target_column": "class",
    },
    "yeast-2_vs_4": {
        "source_file": "imb_IRhigherThan9p1/yeast-2_vs_4/yeast-2_vs_4.zip",
        "loader_type": "KEEL_ZIP",
        "target_column": "class",
    },
    "yeast-2_vs_8": {
        "source_file": "imb_IRhigherThan9p1/yeast-2_vs_8/yeast-2_vs_8.zip",
        "loader_type": "KEEL_ZIP",
        "target_column": "class",
    },
    "yeast1": {
        "source_file": "yeast1.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "Class",
    },
    "yeast4": {
        "source_file": "imb_IRhigherThan9p1/yeast4/yeast4.zip",
        "loader_type": "KEEL_ZIP",
        "target_column": "Class",
    },
    "yeast5": {
        "source_file": "imb_IRhigherThan9p1/yeast5/yeast5.zip",
        "loader_type": "KEEL_ZIP",
        "target_column": "Class",
    },
    "yeast6": {
        "source_file": "imb_IRhigherThan9p1/yeast6/yeast6.zip",
        "loader_type": "KEEL_ZIP",
        "target_column": "Class",
    },

    "ecoli-0-1-3-7_vs_2-6": {
        "source_file": "imb_IRhigherThan9p1/ecoli-0-1-3-7_vs_2-6/ecoli-0-1-3-7_vs_2-6.zip",
        "loader_type": "KEEL_ZIP",
        "target_column": "class",
    },
    "ecoli-0-2-3-4_vs_5": {
        "source_file": "imb_IRhigherThan9p2/ecoli-0-2-3-4_vs_5/ecoli-0-2-3-4_vs_5.zip",
        "loader_type": "KEEL_ZIP",
        "target_column": "class",
    },
    "ecoli-0-3-4_vs_5": {
        "source_file": "imb_IRhigherThan9p2/ecoli-0-3-4_vs_5/ecoli-0-3-4_vs_5.zip",
        "loader_type": "KEEL_ZIP",
        "target_column": "class",
    },
    "ecoli-0-6-7_vs_3-5": {
        "source_file": "imb_IRhigherThan9p2/ecoli-0-6-7_vs_3-5/ecoli-0-6-7_vs_3-5.zip",
        "loader_type": "KEEL_ZIP",
        "target_column": "class",
    },
    "ecoli-0_vs_1": {
        "source_file": "ecoli-0_vs_1.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },
    "ecoli1": {
        "source_file": "ecoli1.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },
    "ecoli2": {
        "source_file": "ecoli2.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },
    "ecoli3": {
        "source_file": "ecoli3.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },
    "ecoli4": {
        "source_file": "ecoli4.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },

    "abalone19": {
        "source_file": "imb_IRhigherThan9p1/abalone19/abalone19.zip",
        "loader_type": "KEEL_ZIP",
        "target_column": "class",
    },
    "abalone9-18": {
        "source_file": "abalone9-18.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },

    "new-thyroid1": {
        "source_file": "new-thyroid1.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },
    "new-thyroid2": {
        "source_file": "newthyroid2.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },

    "page-blocks0": {
        "source_file": "page-blocks0.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },

    "shuttle-c0-vs-c4": {
        "source_file": "shuttle-c0-vs-c4.dat",
        "loader_type": "KEEL_DAT",
        "target_column": "class",
    },
    "shuttle-c2-vs-c4": {
        "source_file": "imb_IRhigherThan9p1/shuttle-c2-vs-c4/shuttle-c2-vs-c4.zip",
        "loader_type": "KEEL_ZIP",
        "target_column": "class",
    },
}


def main():
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(
            "Registry not found: {}".format(
                REGISTRY_FILE
            )
        )

    registry = pd.read_csv(
        REGISTRY_FILE
    )

    if "dataset" not in registry.columns:
        raise ValueError(
            "Registry must contain a dataset column."
        )

    if not BACKUP_FILE.exists():
        shutil.copy2(
            REGISTRY_FILE,
            BACKUP_FILE
        )

    new_columns = [
        "source_file",
        "target_file",
        "loader_type",
    ]

    for column in new_columns:
        if column not in registry.columns:
            registry[column] = ""

    if "separator" not in registry.columns:
        registry["separator"] = ""

    if "header_row" not in registry.columns:
        registry["header_row"] = ""

    if "target_column" not in registry.columns:
        registry["target_column"] = ""

    for dataset, mapping in MAPPINGS.items():
        matches = (
            registry["dataset"].astype(str)
            == dataset
        )

        if not matches.any():
            raise ValueError(
                "Dataset not found in registry: {}".format(
                    dataset
                )
            )

        source_file = mapping["source_file"]
        source_path = RAW_DIR / source_file

        if not source_path.exists():
            raise FileNotFoundError(
                "Source file not found for '{}': {}".format(
                    dataset,
                    source_path
                )
            )

        registry.loc[
            matches,
            "source_file"
        ] = source_file

        registry.loc[
            matches,
            "loader_type"
        ] = mapping["loader_type"]

        registry.loc[
            matches,
            "target_column"
        ] = mapping.get(
            "target_column",
            registry.loc[
                matches,
                "target_column"
            ].iloc[0]
        )

        if "target_file" in mapping:
            registry.loc[
                matches,
                "target_file"
            ] = mapping[
                "target_file"
            ]

        if "separator" in mapping:
            registry.loc[
                matches,
                "separator"
            ] = mapping[
                "separator"
            ]

        if "header_row" in mapping:
            registry.loc[
                matches,
                "header_row"
            ] = mapping[
                "header_row"
            ]

    registry.to_csv(
        REGISTRY_FILE,
        index=False
    )

    print("=" * 80)
    print("VERIFIED DATASET MAPPINGS ADDED")
    print("=" * 80)
    print(
        "Datasets updated : {}".format(
            len(MAPPINGS)
        )
    )
    print(
        "Backup created   : {}".format(
            BACKUP_FILE
        )
    )

    print()
    print(
        registry[
            registry["dataset"].isin(
                MAPPINGS.keys()
            )
        ][
            [
                "dataset",
                "source_file",
                "loader_type",
                "target_column",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
