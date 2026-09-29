
from pathlib import Path
import shutil
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
REGISTRY_FILE = BASE_DIR / "data" / "dataset_registry.csv"
RAW_DIR = BASE_DIR / "data" / "raw"
BACKUP_FILE = BASE_DIR / "data" / "dataset_registry.csv.bak"

UPDATES = {
    "Default of Credit Card Clients": {
        "source_file": "UCI_Credit_Card.csv",
        "loader_type": "CSV",
        "separator": ",",
        "header_row": 0,
        "target_column": "default.payment.next.month",
    },
    "Secondary Mushroom": {
        "source_file": "MushroomDataset/MushroomDataset/secondary_data.csv",
        "loader_type": "CSV",
        "separator": ";",
        "header_row": 0,
        "target_column": "class",
    },
    "Smart Grid Stability": {
        "source_file": "smart_grid_stability_augmented.csv",
        "loader_type": "CSV",
        "separator": ",",
        "header_row": 0,
        "target_column": "stabf",
    },
    "Sonar Mines vs Rocks": {
        "source_file": "sonar.all-data.csv",
        "loader_type": "CSV_HEADERLESS",
        "separator": ",",
        "header_row": -1,
        "target_column": "",
        "target_column_index": 60,
    },
    "Pulsar HTRU2": {
        "source_file": "HTRU_2.csv",
        "loader_type": "CSV_HEADERLESS",
        "separator": ",",
        "header_row": -1,
        "target_column": "",
        "target_column_index": 8,
    },
    "Stroke Prediction": {
        "source_file": "healthcare-dataset-stroke-data.csv",
        "loader_type": "CSV",
        "separator": ",",
        "header_row": 0,
        "target_column": "stroke",
    },
    "IBM Employee Attrition": {
        "source_file": "WA_Fn-UseC_-HR-Employee-Attrition.csv",
        "loader_type": "CSV",
        "separator": ",",
        "header_row": 0,
        "target_column": "Attrition",
    },
    "Credit Card Fraud Detection": {
        "source_file": "creditcard.csv",
        "loader_type": "CSV",
        "separator": ",",
        "header_row": 0,
        "target_column": "Class",
    },
    "Telco Customer Churn": {
        "source_file": "Telco-Customer-Churn.csv",
        "loader_type": "CSV",
        "separator": ",",
        "header_row": 0,
        "target_column": "Churn",
    },
    "Water Potability": {
        "source_file": "water_potability.csv",
        "loader_type": "CSV",
        "separator": ",",
        "header_row": 0,
        "target_column": "Potability",
    },
    "Wine Quality": {
        "source_file": "Wine Quality.csv",
        "loader_type": "CSV",
        "separator": ",",
        "header_row": 0,
        "target_column": "type",
    },
    "Adult": {
        "source_file": "adult.csv",
        "loader_type": "CSV",
        "separator": ",",
        "header_row": 0,
        "target_column": "income",
    },
    "Banknote Authentication": {
        "source_file": "BankNoteAuthentication.csv",
        "loader_type": "CSV",
        "separator": ",",
        "header_row": 0,
        "target_column": "class",
    },
    "Bank Marketing": {
        "source_file": "bank-additional-full.csv",
        "loader_type": "CSV",
        "separator": ";",
        "header_row": 0,
        "target_column": "y",
    },
    "Mushroom Classification": {
        "source_file": "mushrooms.csv",
        "loader_type": "CSV",
        "separator": ",",
        "header_row": 0,
        "target_column": "class",
    },
}


def main():
    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(
            "Registry file not found: {}".format(REGISTRY_FILE)
        )

    registry = pd.read_csv(REGISTRY_FILE)

    if "dataset" not in registry.columns:
        raise ValueError("Registry must contain a 'dataset' column.")

    registry["dataset"] = registry["dataset"].astype(str)

    for column in [
        "source_file",
        "loader_type",
        "separator",
        "header_row",
        "target_column_index",
    ]:
        if column not in registry.columns:
            registry[column] = ""

    if not BACKUP_FILE.exists():
        shutil.copy2(REGISTRY_FILE, BACKUP_FILE)

    missing_registry_datasets = [
        name for name in UPDATES
        if name not in set(registry["dataset"])
    ]

    if missing_registry_datasets:
        raise ValueError(
            "These datasets are missing from the registry: {}".format(
                ", ".join(missing_registry_datasets)
            )
        )

    for dataset, update in UPDATES.items():
        source_path = RAW_DIR / update["source_file"]

        if not source_path.exists():
            raise FileNotFoundError(
                "Source file for '{}' not found: {}".format(
                    dataset, source_path
                )
            )

        mask = registry["dataset"] == dataset

        registry.loc[mask, "source_file"] = update["source_file"]
        registry.loc[mask, "loader_type"] = update["loader_type"]
        registry.loc[mask, "separator"] = update["separator"]
        registry.loc[mask, "header_row"] = str(update["header_row"])
        registry.loc[mask, "target_column"] = update["target_column"]
        registry.loc[mask, "target_column_index"] = str(
            update.get("target_column_index", "")
        )

    registry.to_csv(REGISTRY_FILE, index=False)

    print("=" * 80)
    print("REGISTRY UPDATE COMPLETE")
    print("=" * 80)
    print("Datasets updated :", len(UPDATES))
    print("Backup created   :", BACKUP_FILE)

    print()
    print("UPDATED DATASETS")
    print("-" * 80)

    columns = [
        "dataset",
        "source_file",
        "loader_type",
        "separator",
        "header_row",
        "target_column",
        "target_column_index",
    ]

    print(
        registry.loc[
            registry["dataset"].isin(UPDATES.keys()),
            columns
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
