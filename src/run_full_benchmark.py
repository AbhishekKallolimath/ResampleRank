from pathlib import Path
import argparse
import csv
import io
import math
import re
import time
import zipfile

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
    ADASYN,
)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


RESAMPLING_STRATEGIES = [
    ("none", "No Resampling"),
    ("ros", "Random Oversampling"),
    ("rus", "Random Undersampling"),
    ("smote", "SMOTE"),
    ("adasyn", "ADASYN"),
    ("smote_tomek", "SMOTE-Tomek"),
    ("smote_enn", "SMOTE-ENN"),
]


CLASSIFIERS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
]


N_SPLITS = 5
RANDOM_STATE = 42


def resolve_input_file(input_file):
    if input_file is None:
        return None

    path = Path(input_file)

    if path.is_absolute():
        return path

    return BASE_DIR / path


def parse_keel_attribute_line(line):
    parts = line.strip().split(
        maxsplit=2
    )

    if len(parts) < 3:
        return None, None

    attribute_name = parts[1].strip()

    if (
        attribute_name.startswith("'")
        and attribute_name.endswith("'")
    ):
        attribute_name = attribute_name[1:-1]

    if (
        attribute_name.startswith('"')
        and attribute_name.endswith('"')
    ):
        attribute_name = attribute_name[1:-1]

    definition = parts[2].strip()

    return attribute_name, definition


def read_keel_attribute_definitions(input_file):
    feature_types = {}
    attribute_order = []

    with open(
        input_file,
        "r",
        encoding="utf-8",
    ) as file:

        for raw_line in file:

            line = raw_line.strip()

            if not line:
                continue

            if not line.lower().startswith(
                "@attribute"
            ):
                continue

            name, definition = (
                parse_keel_attribute_line(line)
            )

            if name is None:
                continue

            attribute_order.append(name)

            if (
                definition.startswith("{")
                and definition.endswith("}")
            ):
                feature_types[name] = "categorical"

            elif definition.lower().startswith(
                (
                    "real",
                    "integer",
                    "numeric",
                )
            ):
                feature_types[name] = "numeric"

            else:
                raise ValueError(
                    f"Unsupported KEEL attribute definition "
                    f"for '{name}': {definition}"
                )

    return (
        feature_types,
        attribute_order,
    )

def normalize_name(value):
    value = str(value).lower().strip()

    value = re.sub(
        r"[^a-z0-9]+",
        "",
        value,
    )

    return value

def find_keel_dat_file(dataset_name):
    normalized_dataset = normalize_name(
        dataset_name
    )

    candidates = []

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

        if path.suffix.lower() != ".dat":
            continue

        normalized_stem = normalize_name(
            path.stem
        )

        if normalized_stem == normalized_dataset:
            candidates.append(path)

    if not candidates:
        return None

    candidates.sort(
        key=lambda p: (
            len(p.parts),
            str(p).lower(),
        )
    )

    return candidates[0]


def find_keel_zip_file(dataset_name):
    normalized_dataset = normalize_name(
        dataset_name
    )

    candidates = []

    for path in RAW_DIR.rglob("*.zip"):

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

        normalized_stem = normalize_name(
            path.stem
        )

        normalized_parent = normalize_name(
            path.parent.name
        )

        if (
            normalized_stem
            == normalized_dataset
            or
            normalized_parent
            == normalized_dataset
        ):
            candidates.append(path)

    if not candidates:
        return None

    candidates.sort(
        key=lambda p: (
            0
            if normalize_name(
                p.stem
            )
            == normalized_dataset
            else 1,
            len(p.parts),
            str(p).lower(),
        )
    )

    return candidates[0]


def read_keel_content(
    content,
    source_name,
):
    feature_types_all = {}
    attribute_order = []
    target_column = None

    data_started = False
    records = []

    text = content.decode(
        "utf-8",
        errors="replace",
    )

    for raw_line in text.splitlines():

        line = raw_line.strip()

        if not line:
            continue

        lower = line.lower()

        if lower == "@data":
            data_started = True
            continue

        if not data_started:

            if not lower.startswith(
                "@attribute"
            ):
                continue

            parts = line.split(
                maxsplit=2
            )

            if len(parts) < 3:
                continue

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

            attribute_order.append(
                name
            )

            if (
                definition.startswith("{")
                and definition.endswith("}")
            ):
                feature_types_all[name] = (
                    "categorical"
                )
            elif definition.lower().startswith(
                (
                    "real",
                    "integer",
                    "numeric",
                )
            ):
                feature_types_all[name] = (
                    "numeric"
                )
            else:
                raise ValueError(
                    f"Unsupported KEEL attribute "
                    f"definition in {source_name}: "
                    f"{definition}"
                )

            if name.lower() in {
                "class",
                "target",
                "label",
                "outcome",
            }:
                target_column = name

            continue

        if line.startswith("%"):
            continue

        reader = csv.reader(
            [line],
            skipinitialspace=True,
        )

        row = next(reader)

        row = [
            value.strip()
            for value in row
        ]

        if len(row) != len(
            attribute_order
        ):
            raise ValueError(
                f"Invalid row length in "
                f"{source_name}: expected "
                f"{len(attribute_order)}, found "
                f"{len(row)}"
            )

        records.append(row)

    if not attribute_order:
        raise ValueError(
            f"No KEEL attributes found in "
            f"{source_name}"
        )

    if target_column is None:
        target_column = (
            attribute_order[-1]
        )

    if not records:
        raise ValueError(
            f"No KEEL data rows found in "
            f"{source_name}"
        )

    data = pd.DataFrame(
        records,
        columns=attribute_order,
    )

    feature_columns = [
        column
        for column in attribute_order
        if column != target_column
    ]

    X = data[
        feature_columns
    ].copy()

    y = (
        data[target_column]
        .astype(str)
        .str.strip()
    )

    numeric_features = [
        column
        for column in feature_columns
        if feature_types_all[column]
        == "numeric"
    ]

    categorical_features = [
        column
        for column in feature_columns
        if feature_types_all[column]
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

    return (
        X,
        y,
        numeric_features,
        categorical_features,
        target_column,
    )


def load_keel_dataset(dataset_name):

    dat_file = find_keel_dat_file(
        dataset_name
    )

    if dat_file is not None:

        with open(
            dat_file,
            "rb",
        ) as file:

            content = file.read()

        return read_keel_content(
            content,
            str(
                dat_file.relative_to(
                    BASE_DIR
                )
            ),
        )

    zip_file = find_keel_zip_file(
        dataset_name
    )

    if zip_file is None:
        raise FileNotFoundError(
            f"No KEEL .dat or ZIP file found "
            f"for dataset '{dataset_name}'."
        )

    with zipfile.ZipFile(
        zip_file,
        "r",
    ) as archive:

        members = []

        normalized_dataset = normalize_name(
            dataset_name
        )

        for member in archive.infolist():

            member_name = member.filename

            if member.is_dir():
                continue

            if "__MACOSX/" in member_name:
                continue

            member_base = Path(
                member_name
            ).name

            if member_base.startswith("._"):
                continue

            if not member_base.lower().endswith(
                ".dat"
            ):
                continue

            normalized_member = normalize_name(
                Path(
                    member_base
                ).stem
            )

            if (
                normalized_member
                == normalized_dataset
            ):
                members.append(
                    member
                )

        if not members:

            for member in archive.infolist():

                if member.is_dir():
                    continue

                member_name = member.filename

                if "__MACOSX/" in member_name:
                    continue

                member_base = Path(
                    member_name
                ).name

                if member_base.startswith("._"):
                    continue

                if member_base.lower().endswith(
                    ".dat"
                ):
                    members.append(
                        member
                    )

        if not members:
            raise FileNotFoundError(
                f"No .dat file found inside "
                f"{zip_file}"
            )

        member = members[0]

        content = archive.read(
            member
        )

        source_name = (
            f"{zip_file.relative_to(BASE_DIR)}"
            f" -> {member.filename}"
        )

        return read_keel_content(
            content,
            source_name,
        )


def infer_csv_feature_types(X):
    numeric_features = []
    categorical_features = []

    for column in X.columns:

        series = X[column]

        if pd.api.types.is_numeric_dtype(
            series
        ):
            numeric_features.append(
                column
            )
            continue

        converted = pd.to_numeric(
            series,
            errors="coerce",
        )

        non_missing_original = (
            series.notna().sum()
        )

        non_missing_converted = (
            converted.notna().sum()
        )

        if (
            non_missing_original > 0
            and
            non_missing_original
            == non_missing_converted
        ):
            X[column] = converted
            numeric_features.append(
                column
            )
        else:
            X[column] = (
                series
                .astype(str)
                .str.strip()
            )

            categorical_features.append(
                column
            )

    return (
        numeric_features,
        categorical_features,
    )


def load_csv_dataset(
    dataset_name,
    input_file=None,
    target_column=None,
):
    if input_file is not None:

        feature_file = resolve_input_file(
            input_file
        )

        if not feature_file.exists():
            raise FileNotFoundError(
                f"Input dataset file not found: "
                f"{feature_file}"
            )

        data = pd.read_csv(
            feature_file
        )

        if target_column is None:
            raise ValueError(
                "For a single CSV input file, "
                "--target-column must be provided."
            )

        if target_column not in data.columns:
            raise ValueError(
                f"Target column '{target_column}' "
                f"not found in {feature_file.name}.\n"
                f"Available columns: "
                f"{list(data.columns)}"
            )

        y = (
            data[target_column]
            .astype(str)
            .str.strip()
        )

        X = data.drop(
            columns=[target_column]
        ).copy()

        return (
            X,
            y,
            target_column,
        )

    feature_file = (
        RAW_DIR
        / f"{dataset_name}.csv"
    )

    target_file = (
        RAW_DIR
        / f"{dataset_name}_target.csv"
    )

    if (
        not feature_file.exists()
        or not target_file.exists()
    ):
        raise FileNotFoundError(
            f"Could not find the standardized CSV files "
            f"for dataset '{dataset_name}'.\n\n"
            f"Expected:\n"
            f"  {feature_file}\n"
            f"  {target_file}\n\n"
            f"Alternatively provide:\n"
            f"  --input-file <csv>\n"
            f"  --target-column <column>"
        )

    X = pd.read_csv(
        feature_file
    )

    y_df = pd.read_csv(
        target_file
    )

    if y_df.shape[1] != 1:
        raise ValueError(
            f"Expected one target column in "
            f"{target_file.name}, found "
            f"{y_df.shape[1]}"
        )

    target_column = (
        y_df.columns[0]
    )

    y = (
        y_df.iloc[:, 0]
        .astype(str)
        .str.strip()
    )

    return (
        X,
        y,
        target_column,
    )


def validate_binary_target(y):
    if y.isnull().any():
        raise ValueError(
            "Target contains missing values."
        )

    class_counts = (
        y.value_counts()
    )

    if len(class_counts) != 2:
        raise ValueError(
            "Expected binary classification, "
            f"found {len(class_counts)} classes: "
            f"{list(class_counts.index)}"
        )

    minority_count = int(
        class_counts.min()
    )

    if minority_count < N_SPLITS:
        raise ValueError(
            f"Minority class has only "
            f"{minority_count} samples, but "
            f"{N_SPLITS}-fold stratified CV requires "
            f"at least {N_SPLITS} samples."
        )

    majority_class = (
        class_counts.idxmax()
    )

    minority_class = (
        class_counts.idxmin()
    )

    label_mapping = {
        majority_class: 0,
        minority_class: 1,
    }

    y_encoded = (
        y.map(label_mapping)
        .astype(int)
    )

    return (
        y_encoded,
        majority_class,
        minority_class,
        class_counts,
    )


def finalize_feature_types(
    X,
    numeric_features,
    categorical_features,
):
    X = X.copy()

    missing_columns = [
        column
        for column in X.columns
        if (
            column not in numeric_features
            and
            column not in categorical_features
        )
    ]

    if missing_columns:
        raise ValueError(
            "Could not determine feature type for: "
            f"{missing_columns}"
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

        missing_columns = (
            X.columns[
                X.isnull().any()
            ].tolist()
        )

        raise ValueError(
            f"Missing/invalid feature values in: "
            f"{missing_columns}"
        )

    return X


def load_dataset(
    dataset_name,
    input_file=None,
    target_column=None,
):
    dat_file = find_keel_dat_file(
        dataset_name
    )

    zip_file = find_keel_zip_file(
        dataset_name
    )

    if (
        input_file is None
        and (
            dat_file is not None
            or zip_file is not None
        )
    ):

        (
            X,
            y,
            numeric_features,
            categorical_features,
            detected_target_column,
        ) = load_keel_dataset(
            dataset_name
        )

        target_column = (
            target_column
            or detected_target_column
        )

    else:

        (
            X,
            y,
            detected_target_column,
        ) = load_csv_dataset(
            dataset_name,
            input_file=input_file,
            target_column=target_column,
        )

        target_column = (
            target_column
            or detected_target_column
        )

        (
            numeric_features,
            categorical_features,
        ) = infer_csv_feature_types(
            X
        )

    X = finalize_feature_types(
        X,
        numeric_features,
        categorical_features,
    )

    (
        y,
        majority_class,
        minority_class,
        class_counts,
    ) = validate_binary_target(
        y
    )

    return (
        X,
        y,
        majority_class,
        minority_class,
        numeric_features,
        categorical_features,
        target_column,
        class_counts,
    )
    keel_file = (
        RAW_DIR
        / f"{dataset_name}.dat"
    )

    if (
        input_file is None
        and keel_file.exists()
    ):
        (
            X,
            y,
            numeric_features,
            categorical_features,
            detected_target_column,
        ) = load_keel_dataset(
            dataset_name
        )

        target_column = (
            target_column
            or detected_target_column
        )

    else:
        (
            X,
            y,
            detected_target_column,
        ) = load_csv_dataset(
            dataset_name,
            input_file=input_file,
            target_column=target_column,
        )

        target_column = (
            target_column
            or detected_target_column
        )

        (
            numeric_features,
            categorical_features,
        ) = infer_csv_feature_types(
            X
        )

    X = finalize_feature_types(
        X,
        numeric_features,
        categorical_features,
    )

    (
        y,
        majority_class,
        minority_class,
        class_counts,
    ) = validate_binary_target(
        y
    )

    return (
        X,
        y,
        majority_class,
        minority_class,
        numeric_features,
        categorical_features,
        target_column,
        class_counts,
    )


def build_preprocessor(
    numeric_features,
    categorical_features,
):
    transformers = []

    if numeric_features:

        transformers.append(
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            )
        )

    if categorical_features:

        try:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
        except TypeError:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False,
            )

        transformers.append(
            (
                "categorical",
                encoder,
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError(
            "No numeric or categorical features found."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def get_safe_k_neighbors(y):
    class_counts = (
        y.value_counts()
    )

    minority_count = int(
        class_counts.min()
    )

    minimum_training_minority = (
        minority_count
        - math.ceil(
            minority_count
            / N_SPLITS
        )
    )

    safe_k = min(
        5,
        minimum_training_minority - 1,
    )

    safe_k = max(
        1,
        safe_k,
    )

    return safe_k


def build_classifier(
    classifier_name
):
    if classifier_name == "logistic_regression":

        return LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        )

    if classifier_name == "decision_tree":

        return DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        )

    if classifier_name == "random_forest":

        return RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if classifier_name == "xgboost":

        return XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
        )

    raise ValueError(
        f"Unknown classifier: "
        f"{classifier_name}"
    )


def build_resampler(
    resampling_code,
    y,
):
    k_neighbors = (
        get_safe_k_neighbors(y)
    )

    if resampling_code == "ros":

        return RandomOverSampler(
            random_state=RANDOM_STATE,
        )

    if resampling_code == "rus":

        return RandomUnderSampler(
            random_state=RANDOM_STATE,
        )

    if resampling_code == "smote":

        return SMOTE(
            random_state=RANDOM_STATE,
            k_neighbors=k_neighbors,
        )

    if resampling_code == "adasyn":

        return ADASYN(
            random_state=RANDOM_STATE,
            n_neighbors=k_neighbors,
        )

    if resampling_code == "smote_tomek":

        return SMOTETomek(
            random_state=RANDOM_STATE,
            smote=SMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=k_neighbors,
            ),
        )

    if resampling_code == "smote_enn":

        return SMOTEENN(
            random_state=RANDOM_STATE,
            smote=SMOTE(
                random_state=RANDOM_STATE,
                k_neighbors=k_neighbors,
            ),
        )

    if resampling_code == "none":
        return None

    raise ValueError(
        f"Unknown resampling strategy: "
        f"{resampling_code}"
    )


def build_pipeline(
    resampling_code,
    classifier_name,
    numeric_features,
    categorical_features,
    y,
):
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    classifier = build_classifier(
        classifier_name
    )

    steps = [
        (
            "preprocessor",
            preprocessor,
        )
    ]

    if resampling_code != "none":

        resampler = build_resampler(
            resampling_code,
            y,
        )

        steps.append(
            (
                "resampler",
                resampler,
            )
        )

    steps.append(
        (
            "classifier",
            classifier,
        )
    )

    return ImbPipeline(
        steps=steps
    )


def calculate_macro_f1_mathematically(cm):
    if cm.shape != (2, 2):
        raise ValueError(
            "Expected a 2x2 confusion matrix."
        )

    tn, fp, fn, tp = cm.ravel()

    def calculate_f1(
        tp_value,
        fp_value,
        fn_value,
    ):
        precision_denominator = (
            tp_value
            + fp_value
        )

        recall_denominator = (
            tp_value
            + fn_value
        )

        precision = (
            tp_value
            / precision_denominator
            if precision_denominator != 0
            else 0.0
        )

        recall = (
            tp_value
            / recall_denominator
            if recall_denominator != 0
            else 0.0
        )

        if (
            precision
            + recall
            == 0
        ):
            return 0.0

        return (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    f1_class_0 = calculate_f1(
        tn,
        fn,
        fp,
    )

    f1_class_1 = calculate_f1(
        tp,
        fp,
        fn,
    )

    macro_f1 = (
        f1_class_0
        + f1_class_1
    ) / 2

    return (
        f1_class_0,
        f1_class_1,
        macro_f1,
    )


def run_classifier_benchmark(
    dataset_name,
    X,
    y,
    numeric_features,
    categorical_features,
    classifier_name,
):
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    safe_k = (
        get_safe_k_neighbors(y)
    )

    print("\n")
    print("=" * 75)
    print(
        f"{dataset_name} - "
        f"{classifier_name}"
    )
    print("=" * 75)

    print(
        f"\nNumeric features     : "
        f"{len(numeric_features)}"
    )

    print(
        f"Categorical features : "
        f"{len(categorical_features)}"
    )

    print(
        f"Safe neighbor k      : "
        f"{safe_k}"
    )

    for (
        strategy_code,
        strategy_name,
    ) in RESAMPLING_STRATEGIES:

        print(
            f"\nRunning: "
            f"{strategy_name}"
        )

        model = build_pipeline(
            strategy_code,
            classifier_name,
            numeric_features,
            categorical_features,
            y,
        )

        start_time = (
            time.perf_counter()
        )

        try:

            y_pred = cross_val_predict(
                model,
                X,
                y,
                cv=cv,
                method="predict",
            )

            elapsed_time = (
                time.perf_counter()
                - start_time
            )

            cm = confusion_matrix(
                y,
                y_pred,
                labels=[0, 1],
            )

            tn, fp, fn, tp = (
                cm.ravel()
            )

            accuracy = accuracy_score(
                y,
                y_pred,
            )

            balanced_accuracy = (
                balanced_accuracy_score(
                    y,
                    y_pred,
                )
            )

            precision = precision_score(
                y,
                y_pred,
                average="binary",
                zero_division=0,
            )

            recall = recall_score(
                y,
                y_pred,
                average="binary",
                zero_division=0,
            )

            macro_f1_code = f1_score(
                y,
                y_pred,
                average="macro",
                zero_division=0,
            )

            (
                f1_class_0_math,
                f1_class_1_math,
                macro_f1_math,
            ) = calculate_macro_f1_mathematically(
                cm
            )

            metric_error = abs(
                macro_f1_math
                - macro_f1_code
            )

            verification = (
                "PASS"
                if metric_error < 1e-6
                else "FAIL"
            )

            failure_reason = ""

            print(
                f"Macro-F1          : "
                f"{macro_f1_math:.6f}"
            )

            print(
                f"Verification      : "
                f"{verification}"
            )

            print(
                f"Execution Time    : "
                f"{elapsed_time:.6f} seconds"
            )

        except Exception as error:

            elapsed_time = (
                time.perf_counter()
                - start_time
            )

            if strategy_code != "none":

                verification = (
                    "NOT_APPLICABLE"
                )

                failure_reason = str(
                    error
                )

                tn = float("nan")
                fp = float("nan")
                fn = float("nan")
                tp = float("nan")

                accuracy = float("nan")
                balanced_accuracy = float(
                    "nan"
                )
                precision = float("nan")
                recall = float("nan")

                f1_class_0_math = float(
                    "nan"
                )
                f1_class_1_math = float(
                    "nan"
                )
                macro_f1_code = float(
                    "nan"
                )
                macro_f1_math = float(
                    "nan"
                )
                metric_error = float(
                    "nan"
                )

                print(
                    "Status            : "
                    "NOT_APPLICABLE"
                )

                print(
                    f"Reason            : "
                    f"{failure_reason}"
                )

                print(
                    f"Execution Time    : "
                    f"{elapsed_time:.6f} seconds"
                )

            else:
                raise

        results.append(
            {
                "dataset": dataset_name,
                "resampling_code": strategy_code,
                "resampling": strategy_name,
                "classifier": classifier_name,
                "samples": X.shape[0],
                "features": X.shape[1],
                "numeric_features": len(
                    numeric_features
                ),
                "categorical_features": len(
                    categorical_features
                ),
                "safe_k_neighbors": safe_k,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "precision": precision,
                "recall": recall,
                "f1_class_0_math": (
                    f1_class_0_math
                ),
                "f1_class_1_math": (
                    f1_class_1_math
                ),
                "macro_f1_code": (
                    macro_f1_code
                ),
                "macro_f1_math": (
                    macro_f1_math
                ),
                "metric_error": (
                    metric_error
                ),
                "verification": (
                    verification
                ),
                "failure_reason": (
                    failure_reason
                ),
                "execution_time_seconds": (
                    elapsed_time
                ),
            }
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            by="macro_f1_math",
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )

    result_file = (
        RESULTS_DIR
        / f"{dataset_name}_"
        f"{classifier_name}_results.csv"
    )

    results_df.to_csv(
        result_file,
        index=False,
    )

    print(
        f"\nSaved: {result_file}"
    )

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete ResampleRank "
            "benchmark for one dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Dataset name, for example "
            "pima, glass0, banknote, etc."
        ),
    )

    parser.add_argument(
        "--input-file",
        default=None,
        help=(
            "Optional CSV file. Use this when "
            "the dataset is stored as a single "
            "CSV containing both features and target."
        ),
    )

    parser.add_argument(
        "--target-column",
        default=None,
        help=(
            "Target column name for a single CSV "
            "input file."
        ),
    )

    args = parser.parse_args()

    dataset_name = (
        args.dataset.strip()
    )

    (
        X,
        y,
        majority_class,
        minority_class,
        numeric_features,
        categorical_features,
        target_column,
        class_counts,
    ) = load_dataset(
        dataset_name,
        input_file=args.input_file,
        target_column=args.target_column,
    )

    print("=" * 75)
    print(
        "ResampleRank - Full Dataset Benchmark"
    )
    print("=" * 75)

    print(
        f"\nDataset           : "
        f"{dataset_name}"
    )

    print(
        f"Input target      : "
        f"{target_column}"
    )

    print(
        f"Samples           : "
        f"{X.shape[0]}"
    )

    print(
        f"Features          : "
        f"{X.shape[1]}"
    )

    print(
        f"Numeric features  : "
        f"{len(numeric_features)}"
    )

    print(
        f"Categorical       : "
        f"{len(categorical_features)}"
    )

    print(
        f"Majority class    : "
        f"{majority_class} -> 0"
    )

    print(
        f"Minority class    : "
        f"{minority_class} -> 1"
    )

    print(
        "\nClass distribution:"
    )

    print(
        class_counts.to_string()
    )

    classifiers = CLASSIFIERS

    all_results = []

    for classifier_name in classifiers:

        result_df = run_classifier_benchmark(
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
        RESULTS_DIR
        / f"{dataset_name}_all_results.csv"
    )

    combined.to_csv(
        combined_file,
        index=False,
    )

    successful_experiments = (
        combined["verification"]
        == "PASS"
    ).sum()

    verification_failures = (
        combined["verification"]
        == "FAIL"
    ).sum()

    not_applicable = (
        combined["verification"]
        == "NOT_APPLICABLE"
    ).sum()

    valid_macro_f1 = (
        combined.loc[
            combined["verification"]
            == "PASS",
            "macro_f1_math",
        ]
    )

    print("\n")
    print("=" * 75)
    print("BENCHMARK SUMMARY")
    print("=" * 75)

    print(
        f"Total experiments       : "
        f"{len(combined)}"
    )

    print(
        f"Successful experiments  : "
        f"{successful_experiments}"
    )

    print(
        f"Verification failures  : "
        f"{verification_failures}"
    )

    print(
        f"Not applicable          : "
        f"{not_applicable}"
    )

    if not valid_macro_f1.empty:

        print(
            f"Macro-F1 minimum       : "
            f"{valid_macro_f1.min():.6f}"
        )

        print(
            f"Macro-F1 maximum       : "
            f"{valid_macro_f1.max():.6f}"
        )

    else:

        print(
            "Macro-F1 minimum       : N/A"
        )

        print(
            "Macro-F1 maximum       : N/A"
        )

    print(
        "\nCombined results saved to:"
    )

    print(
        combined_file
    )


if __name__ == "__main__":
    main()