from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
INPUT_PATH = (
    RESULTS_DIR
    / "generalization_comparison"
    / "lodo_vs_nested.csv"
)

OUTPUT_DIR = (
    RESULTS_DIR
    / "generalization_comparison"
)

FIGURES_DIR = Path("figures")


def load_results():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Required file not found: {INPUT_PATH}"
        )

    return pd.read_csv(INPUT_PATH)


def normalize_accuracy(value):
    value = float(value)

    if value <= 1.0:
        return value * 100.0

    return value


def create_summary_table(df):
    rows = []

    for classifier in df["classifier"].unique():
        row = df[
            df["classifier"] == classifier
        ].iloc[0]

        rows.append({
            "classifier": classifier,

            "LODO_MAE": row["lodo_mae"],
            "Nested_MAE": row["nested_mae"],

            "LODO_RMSE": row["lodo_rmse"],
            "Nested_RMSE": row["nested_rmse"],

            "LODO_Selected_Improvement": (
                row["lodo_selected_improvement"]
            ),
            "Nested_Selected_Improvement": (
                row["nested_selected_improvement"]
            ),

            "LODO_Regret": row["lodo_regret"],
            "Nested_Regret": row["nested_regret"],

            "LODO_Top1_Percent": (
                normalize_accuracy(
                    row["lodo_top1_accuracy"]
                )
            ),
            "Nested_Top1_Percent": (
                normalize_accuracy(
                    row["nested_top1_accuracy"]
                )
            ),

            "LODO_Top3_Percent": (
                normalize_accuracy(
                    row["lodo_top3_accuracy"]
                )
            ),
            "Nested_Top3_Percent": (
                normalize_accuracy(
                    row["nested_top3_accuracy"]
                )
            ),
        })

    return pd.DataFrame(rows)


def create_protocol_summary(df):
    lodo = {
        "protocol": "LODO",
        "MAE": df["lodo_mae"].mean(),
        "RMSE": df["lodo_rmse"].mean(),
        "Selected_Improvement": (
            df["lodo_selected_improvement"].mean()
        ),
        "Regret": df["lodo_regret"].mean(),
        "Top1_Percent": (
            df["lodo_top1_accuracy"]
            .apply(normalize_accuracy)
            .mean()
        ),
        "Top3_Percent": (
            df["lodo_top3_accuracy"]
            .apply(normalize_accuracy)
            .mean()
        )
    }

    nested = {
        "protocol": "Nested",
        "MAE": df["nested_mae"].mean(),
        "RMSE": df["nested_rmse"].mean(),
        "Selected_Improvement": (
            df["nested_selected_improvement"].mean()
        ),
        "Regret": df["nested_regret"].mean(),
        "Top1_Percent": (
            df["nested_top1_accuracy"]
            .apply(normalize_accuracy)
            .mean()
        ),
        "Top3_Percent": (
            df["nested_top3_accuracy"]
            .apply(normalize_accuracy)
            .mean()
        )
    }

    return pd.DataFrame(
        [lodo, nested]
    )


def create_metric_difference(df):
    rows = []

    mappings = [
        ("MAE", "lodo_mae", "nested_mae"),
        ("RMSE", "lodo_rmse", "nested_rmse"),
        (
            "Selected Improvement",
            "lodo_selected_improvement",
            "nested_selected_improvement"
        ),
        (
            "Regret",
            "lodo_regret",
            "nested_regret"
        )
    ]

    for classifier in df["classifier"].unique():
        row = df[
            df["classifier"] == classifier
        ].iloc[0]

        result = {
            "classifier": classifier
        }

        for name, lodo_col, nested_col in mappings:
            lodo_value = float(
                row[lodo_col]
            )

            nested_value = float(
                row[nested_col]
            )

            key = (
                name.lower()
                .replace(" ", "_")
            )

            result[
                f"{key}_difference_nested_minus_lodo"
            ] = (
                nested_value
                - lodo_value
            )

        result[
            "top1_difference_nested_minus_lodo"
        ] = (
            normalize_accuracy(
                row["nested_top1_accuracy"]
            )
            - normalize_accuracy(
                row["lodo_top1_accuracy"]
            )
        )

        result[
            "top3_difference_nested_minus_lodo"
        ] = (
            normalize_accuracy(
                row["nested_top3_accuracy"]
            )
            - normalize_accuracy(
                row["lodo_top3_accuracy"]
            )
        )

        rows.append(result)

    return pd.DataFrame(rows)


def create_figure(df):
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    classifiers = df[
        "classifier"
    ].tolist()

    x = range(
        len(classifiers)
    )

    lodo_mae = df[
        "lodo_mae"
    ].tolist()

    nested_mae = df[
        "nested_mae"
    ].tolist()

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    positions_lodo = [
        value - width / 2
        for value in x
    ]

    positions_nested = [
        value + width / 2
        for value in x
    ]

    ax.bar(
        positions_lodo,
        lodo_mae,
        width,
        label="LODO"
    )

    ax.bar(
        positions_nested,
        nested_mae,
        width,
        label="Nested Train/Validation/Test"
    )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        classifiers,
        rotation=20,
        ha="right"
    )

    ax.set_ylabel(
        "Mean Absolute Error"
    )

    ax.set_title(
        "LODO vs Nested Generalization: Meta-Learning MAE"
    )

    ax.legend()

    fig.tight_layout()

    png_path = (
        FIGURES_DIR
        / "figure_09_lodo_vs_nested_mae.png"
    )

    pdf_path = (
        FIGURES_DIR
        / "figure_09_lodo_vs_nested_mae.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {png_path}"
    )

    print(
        f"Saved: {pdf_path}"
    )


def main():
    print("=" * 75)
    print(
        "ResampleRank - LODO vs Nested Comparison"
    )
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_results()

    classifier_table = (
        create_summary_table(df)
    )

    protocol_table = (
        create_protocol_summary(df)
    )

    difference_table = (
        create_metric_difference(df)
    )

    classifier_path = (
        OUTPUT_DIR
        / "lodo_vs_nested_paper_table.csv"
    )

    protocol_path = (
        OUTPUT_DIR
        / "protocol_paper_summary.csv"
    )

    difference_path = (
        OUTPUT_DIR
        / "protocol_metric_differences.csv"
    )

    classifier_table.to_csv(
        classifier_path,
        index=False
    )

    protocol_table.to_csv(
        protocol_path,
        index=False
    )

    difference_table.to_csv(
        difference_path,
        index=False
    )

    create_figure(df)

    print(
        "\n" + "=" * 75
    )

    print(
        "PROTOCOL COMPARISON SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        protocol_table.to_string(
            index=False
        )
    )

    print(
        "\nSaved tables:"
    )

    print(
        classifier_path.resolve()
    )

    print(
        protocol_path.resolve()
    )

    print(
        difference_path.resolve()
    )


if __name__ == "__main__":
    main()