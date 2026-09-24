from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")

LODO_DIR = RESULTS_DIR / "meta_model"
NESTED_DIR = RESULTS_DIR / "nested_meta_model"

SELECTION_DIR = (
    RESULTS_DIR / "selection_performance"
)

OUTPUT_DIR = (
    RESULTS_DIR / "generalization_comparison"
)

META_MODELS = [
    "ridge",
    "random_forest",
    "gradient_boosting",
    "xgboost",
]

CLASSIFIERS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
]

METRICS = [
    "mae",
    "rmse",
]


def safe_mean(values):
    values = pd.to_numeric(
        values,
        errors="coerce"
    )

    values = values.dropna()

    if len(values) == 0:
        return np.nan

    return float(values.mean())


def load_lodo_prediction_metrics():
    rows = []

    for meta_model in META_MODELS:

        for classifier in CLASSIFIERS:

            filename = (
                f"{meta_model}_"
                f"{classifier}_predictions.csv"
            )

            path = LODO_DIR / filename

            if not path.exists():
                print(
                    f"WARNING: Missing LODO file: {path}"
                )
                continue

            df = pd.read_csv(path)

            required = [
                "delta_macro_f1_vs_baseline",
                "predicted_delta_macro_f1"
            ]

            missing = [
                column
                for column in required
                if column not in df.columns
            ]

            if missing:
                print(
                    f"WARNING: Missing columns in {path}: "
                    f"{missing}"
                )
                continue

            actual = pd.to_numeric(
                df[
                    "delta_macro_f1_vs_baseline"
                ],
                errors="coerce"
            )

            predicted = pd.to_numeric(
                df[
                    "predicted_delta_macro_f1"
                ],
                errors="coerce"
            )

            valid = (
                actual.notna()
                & predicted.notna()
            )

            actual = actual[valid]
            predicted = predicted[valid]

            if len(actual) == 0:
                continue

            error = (
                predicted.to_numpy()
                - actual.to_numpy()
            )

            mae = float(
                np.mean(
                    np.abs(error)
                )
            )

            rmse = float(
                np.sqrt(
                    np.mean(
                        error ** 2
                    )
                )
            )

            rows.append({
                "meta_model": meta_model,
                "classifier": classifier,
                "rows": len(actual),
                "mae": mae,
                "rmse": rmse
            })

    return pd.DataFrame(rows)


def aggregate_lodo_metrics(df):
    if len(df) == 0:
        return pd.DataFrame()

    result = (
        df.groupby(
            "classifier",
            as_index=False
        )
        .agg(
            lodo_mae=(
                "mae",
                "mean"
            ),
            lodo_rmse=(
                "rmse",
                "mean"
            ),
            lodo_meta_models=(
                "meta_model",
                "nunique"
            )
        )
    )

    return result


def load_lodo_selection_metrics():
    path = (
        SELECTION_DIR
        / "selection_performance_summary.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing selection summary: {path}"
        )

    df = pd.read_csv(path)

    required = [
        "meta_model",
        "classifier",
        "mean_selected_improvement",
        "mean_oracle_improvement",
        "mean_gap_to_oracle",
        "mean_regret",
        "top1_accuracy",
        "top3_accuracy",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing LODO selection columns: "
            + ", ".join(missing)
        )

    return df


def aggregate_lodo_selection_metrics(df):
    result = (
        df.groupby(
            "classifier",
            as_index=False
        )
        .agg(
            lodo_selected_improvement=(
                "mean_selected_improvement",
                "mean"
            ),
            lodo_oracle_improvement=(
                "mean_oracle_improvement",
                "mean"
            ),
            lodo_gap_to_oracle=(
                "mean_gap_to_oracle",
                "mean"
            ),
            lodo_regret=(
                "mean_regret",
                "mean"
            ),
            lodo_top1_accuracy=(
                "top1_accuracy",
                "mean"
            ),
            lodo_top3_accuracy=(
                "top3_accuracy",
                "mean"
            ),
        )
    )

    return result


def load_nested_metrics():
    path = (
        NESTED_DIR
        / "nested_test_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing nested prediction file: {path}"
        )

    df = pd.read_csv(path)

    required = [
        "classifier",
        "delta_macro_f1_vs_baseline",
        "predicted_delta_macro_f1"
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing nested prediction columns: "
            + ", ".join(missing)
        )

    rows = []

    for classifier in CLASSIFIERS:

        subset = df[
            df["classifier"] == classifier
        ].copy()

        actual = pd.to_numeric(
            subset[
                "delta_macro_f1_vs_baseline"
            ],
            errors="coerce"
        )

        predicted = pd.to_numeric(
            subset[
                "predicted_delta_macro_f1"
            ],
            errors="coerce"
        )

        valid = (
            actual.notna()
            & predicted.notna()
        )

        actual = actual[valid]
        predicted = predicted[valid]

        if len(actual) == 0:
            continue

        error = (
            predicted.to_numpy()
            - actual.to_numpy()
        )

        rows.append({
            "classifier": classifier,
            "nested_test_rows": len(actual),
            "nested_mae": float(
                np.mean(
                    np.abs(error)
                )
            ),
            "nested_rmse": float(
                np.sqrt(
                    np.mean(
                        error ** 2
                    )
                )
            )
        })

    return pd.DataFrame(rows)


def load_nested_selection_metrics():
    path = (
        NESTED_DIR
        / "nested_test_selection_results.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing nested selection file: {path}"
        )

    df = pd.read_csv(path)

    required = [
        "classifier",
        "selected_improvement",
        "oracle_improvement",
        "regret",
        "top1_accuracy",
        "top3_accuracy",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing nested selection columns: "
            + ", ".join(missing)
        )

    result = (
        df.groupby(
            "classifier",
            as_index=False
        )
        .agg(
            nested_selected_improvement=(
                "selected_improvement",
                "mean"
            ),
            nested_oracle_improvement=(
                "oracle_improvement",
                "mean"
            ),
            nested_gap_to_oracle=(
                "regret",
                "mean"
            ),
            nested_regret=(
                "regret",
                "mean"
            ),
            nested_top1_accuracy=(
                "top1_accuracy",
                "mean"
            ),
            nested_top3_accuracy=(
                "top3_accuracy",
                "mean"
            ),
        )
    )

    return result


def create_classifier_comparison():

    lodo_predictions = (
        load_lodo_prediction_metrics()
    )

    lodo_prediction_summary = (
        aggregate_lodo_metrics(
            lodo_predictions
        )
    )

    lodo_selection = (
        load_lodo_selection_metrics()
    )

    lodo_selection_summary = (
        aggregate_lodo_selection_metrics(
            lodo_selection
        )
    )

    nested_predictions = (
        load_nested_metrics()
    )

    nested_selection = (
        load_nested_selection_metrics()
    )

    result = pd.merge(
        lodo_prediction_summary,
        lodo_selection_summary,
        on="classifier",
        how="outer"
    )

    result = pd.merge(
        result,
        nested_predictions,
        on="classifier",
        how="outer"
    )

    result = pd.merge(
        result,
        nested_selection,
        on="classifier",
        how="outer"
    )

    return result


def create_protocol_summary(
    classifier_comparison
):

    metric_columns = [
        "lodo_mae",
        "lodo_rmse",
        "lodo_selected_improvement",
        "lodo_oracle_improvement",
        "lodo_gap_to_oracle",
        "lodo_regret",
        "lodo_top1_accuracy",
        "lodo_top3_accuracy",
        "nested_mae",
        "nested_rmse",
        "nested_selected_improvement",
        "nested_oracle_improvement",
        "nested_gap_to_oracle",
        "nested_regret",
        "nested_top1_accuracy",
        "nested_top3_accuracy",
    ]

    rows = []

    for protocol, prefix in [
        ("LODO", "lodo"),
        ("Nested", "nested")
    ]:

        row = {
            "protocol": protocol
        }

        for metric in metric_columns:
            if metric.startswith(prefix):
                row[metric.replace(
                    f"{prefix}_",
                    ""
                )] = safe_mean(
                    classifier_comparison[
                        metric
                    ]
                )

        rows.append(row)

    return pd.DataFrame(rows)


def create_metric_difference_table(
    classifier_comparison
):

    rows = []

    mappings = [
        (
            "MAE",
            "lodo_mae",
            "nested_mae"
        ),
        (
            "RMSE",
            "lodo_rmse",
            "nested_rmse"
        ),
        (
            "Selected Improvement",
            "lodo_selected_improvement",
            "nested_selected_improvement"
        ),
        (
            "Oracle Improvement",
            "lodo_oracle_improvement",
            "nested_oracle_improvement"
        ),
        (
            "Gap to Oracle",
            "lodo_gap_to_oracle",
            "nested_gap_to_oracle"
        ),
        (
            "Regret",
            "lodo_regret",
            "nested_regret"
        ),
        (
            "Top-1 Accuracy",
            "lodo_top1_accuracy",
            "nested_top1_accuracy"
        ),
        (
            "Top-3 Accuracy",
            "lodo_top3_accuracy",
            "nested_top3_accuracy"
        ),
    ]

    for classifier in CLASSIFIERS:

        subset = classifier_comparison[
            classifier_comparison[
                "classifier"
            ] == classifier
        ]

        if len(subset) == 0:
            continue

        row = {
            "classifier": classifier
        }

        record = subset.iloc[0]

        for metric_name, lodo_col, nested_col in mappings:

            lodo_value = record.get(
                lodo_col,
                np.nan
            )

            nested_value = record.get(
                nested_col,
                np.nan
            )

            row[
                f"lodo_{metric_name.lower().replace(' ', '_')}"
            ] = lodo_value

            row[
                f"nested_{metric_name.lower().replace(' ', '_')}"
            ] = nested_value

            row[
                f"difference_{metric_name.lower().replace(' ', '_')}"
            ] = (
                nested_value
                - lodo_value
                if pd.notna(nested_value)
                and pd.notna(lodo_value)
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def main():

    print("=" * 75)
    print(
        "ResampleRank - Generalization Protocol Comparison"
    )
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\nComparing:"
    )

    print(
        "1. Leave-One-Dataset-Out (LODO)"
    )

    print(
        "2. Nested Train / Validation / Test"
    )

    print(
        "\nImportant:"
    )

    print(
        "LODO metrics are averaged across the "
        "four fixed meta-models."
    )

    print(
        "Nested metrics use the model selected "
        "from validation in each fold."
    )

    comparison = (
        create_classifier_comparison()
    )

    comparison_path = (
        OUTPUT_DIR
        / "lodo_vs_nested.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False
    )

    protocol_summary = (
        create_protocol_summary(
            comparison
        )
    )

    protocol_summary_path = (
        OUTPUT_DIR
        / "protocol_summary.csv"
    )

    protocol_summary.to_csv(
        protocol_summary_path,
        index=False
    )

    metric_difference = (
        create_metric_difference_table(
            comparison
        )
    )

    difference_path = (
        OUTPUT_DIR
        / "metric_differences.csv"
    )

    metric_difference.to_csv(
        difference_path,
        index=False
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "GENERALIZATION COMPARISON"
    )

    print(
        "=" * 75
    )

    display_columns = [
        "classifier",
        "lodo_mae",
        "nested_mae",
        "lodo_rmse",
        "nested_rmse",
        "lodo_selected_improvement",
        "nested_selected_improvement",
        "lodo_top1_accuracy",
        "nested_top1_accuracy",
        "lodo_top3_accuracy",
        "nested_top3_accuracy",
        "lodo_regret",
        "nested_regret",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in comparison.columns
    ]

    print(
        comparison[
            available_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nProtocol summary:"
    )

    print(
        protocol_summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved files:"
    )

    print(
        comparison_path.resolve()
    )

    print(
        protocol_summary_path.resolve()
    )

    print(
        difference_path.resolve()
    )


if __name__ == "__main__":
    main()