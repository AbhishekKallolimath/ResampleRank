from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

META_DATASET_PATH = RESULTS_DIR / "meta_dataset.csv"
META_MODEL_OVERALL_PATH = RESULTS_DIR / "meta_model" / "meta_model_overall.csv"
COMPUTATIONAL_SAVINGS_PATH = (
    RESULTS_DIR / "computational_savings" / "computational_savings_summary.csv"
)
STATISTICAL_FRIEDMAN_PATH = (
    RESULTS_DIR / "statistical_analysis" / "friedman_results.csv"
)
STATISTICAL_EFFECTS_PATH = (
    RESULTS_DIR / "statistical_analysis" / "effect_sizes.csv"
)
PREDICTIONS_DIR = RESULTS_DIR / "meta_model"


def save_figure(fig, filename):
    png_path = FIGURES_DIR / f"{filename}.png"
    pdf_path = FIGURES_DIR / f"{filename}.pdf"

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

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def find_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column

    lowered = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        key = str(candidate).lower()

        if key in lowered:
            return lowered[key]

    for column in df.columns:
        column_text = str(column).lower()

        for candidate in candidates:
            if str(candidate).lower() in column_text:
                return column

    return None


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return pd.read_csv(path)


def figure_01_dataset_imbalance():
    df = load_csv(META_DATASET_PATH)

    columns = [
        "dataset",
        "imbalance_ratio"
    ]

    data = (
        df[columns]
        .drop_duplicates()
        .sort_values(
            "imbalance_ratio",
            ascending=True
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        data["dataset"],
        data["imbalance_ratio"]
    )

    ax.set_xlabel("Imbalance Ratio")
    ax.set_ylabel("Dataset")
    ax.set_title(
        "Class Imbalance Across Primary Datasets"
    )

    for index, value in enumerate(
        data["imbalance_ratio"]
    ):
        ax.text(
            value,
            index,
            f" {value:.2f}",
            va="center",
            fontsize=9
        )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_01_dataset_imbalance"
    )


def figure_02_strategy_macro_f1():
    df = load_csv(META_DATASET_PATH)

    grouped = (
        df.groupby(
            "resampling"
        )["target_macro_f1"]
        .agg(
            ["mean", "std"]
        )
        .sort_values(
            "mean",
            ascending=True
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        grouped.index,
        grouped["mean"],
        xerr=grouped["std"],
        capsize=4
    )

    ax.set_xlabel("Mean Macro-F1")
    ax.set_ylabel("Resampling Strategy")
    ax.set_title(
        "Macro-F1 Across Resampling Strategies"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_02_strategy_macro_f1"
    )


def figure_03_dataset_macro_f1():
    df = load_csv(META_DATASET_PATH)

    grouped = (
        df.groupby(
            "dataset"
        )["target_macro_f1"]
        .mean()
        .sort_values(
            ascending=True
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        grouped.index,
        grouped.values
    )

    ax.set_xlabel(
        "Mean Macro-F1"
    )

    ax.set_ylabel(
        "Dataset"
    )

    ax.set_title(
        "Mean Macro-F1 Across Primary Datasets"
    )

    ax.set_xlim(
        0,
        min(
            1.05,
            max(grouped.values) + 0.05
        )
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_03_dataset_macro_f1"
    )


def figure_04_meta_model_error():
    df = load_csv(
        META_MODEL_OVERALL_PATH
    )

    models = df["model"].tolist()
    mae = df["mae"].tolist()
    rmse = df["rmse"].tolist()

    x = np.arange(
        len(models)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width / 2,
        mae,
        width,
        label="MAE"
    )

    ax.bar(
        x + width / 2,
        rmse,
        width,
        label="RMSE"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        models,
        rotation=20,
        ha="right"
    )

    ax.set_ylabel("Error")
    ax.set_title(
        "Meta-Learning Prediction Error"
    )

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig,
        "figure_04_meta_model_error"
    )


def figure_05_ranking_accuracy():
    df = load_csv(
        META_MODEL_OVERALL_PATH
    )

    models = df["model"].tolist()

    top1 = (
        df["top1_resampling_accuracy"] * 100
    )

    top3 = (
        df["top3_resampling_accuracy"] * 100
    )

    x = np.arange(
        len(models)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width / 2,
        top1,
        width,
        label="Top-1"
    )

    ax.bar(
        x + width / 2,
        top3,
        width,
        label="Top-3"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        models,
        rotation=20,
        ha="right"
    )

    ax.set_ylabel(
        "Accuracy (%)"
    )

    ax.set_title(
        "Resampling Ranking Accuracy"
    )

    ax.set_ylim(
        0,
        100
    )

    ax.legend()

    fig.tight_layout()

    save_figure(
        fig,
        "figure_05_ranking_accuracy"
    )


def figure_06_computational_savings():
    df = load_csv(
        COMPUTATIONAL_SAVINGS_PATH
    )

    grouped = (
        df.groupby(
            "meta_model"
        )["mean_savings_percent"]
        .mean()
        .sort_values(
            ascending=True
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        grouped.index,
        grouped.values
    )

    ax.set_xlabel(
        "Mean Computational Savings (%)"
    )

    ax.set_ylabel(
        "Meta-Model"
    )

    ax.set_title(
        "Computational Savings from Strategy Selection"
    )

    ax.set_xlim(
        0,
        100
    )

    for index, value in enumerate(
        grouped.values
    ):
        ax.text(
            value,
            index,
            f" {value:.2f}%",
            va="center"
        )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_06_computational_savings"
    )


def get_prediction_columns(df):
    actual_column = "delta_macro_f1_vs_baseline"
    predicted_column = "predicted_delta_macro_f1"

    if (
        actual_column in df.columns
        and predicted_column in df.columns
    ):
        return actual_column, predicted_column

    return None, None


def figure_07_predicted_vs_actual():
    files = sorted(
        PREDICTIONS_DIR.glob(
            "*_predictions.csv"
        )
    )

    if not files:
        print(
            "WARNING: No prediction files found. "
            "Skipping Figure 7."
        )
        return

    frames = []

    for path in files:
        try:
            df = pd.read_csv(path)

            actual_column, predicted_column = (
                get_prediction_columns(df)
            )

            if (
                actual_column is None
                or predicted_column is None
            ):
                print(
                    f"WARNING: Required columns not found in {path}"
                )
                continue

            temp = pd.DataFrame({
                "actual": pd.to_numeric(
                    df[actual_column],
                    errors="coerce"
                ),
                "predicted": pd.to_numeric(
                    df[predicted_column],
                    errors="coerce"
                )
            })

            temp = temp.dropna()

            if len(temp) > 0:
                frames.append(temp)

        except Exception as exc:
            print(
                f"WARNING: Could not read {path}: {exc}"
            )

    if not frames:
        print(
            "WARNING: No valid prediction data found. "
            "Skipping Figure 7."
        )
        return

    data = pd.concat(
        frames,
        ignore_index=True
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    ax.scatter(
        data["actual"],
        data["predicted"],
        alpha=0.6
    )

    minimum = min(
        data["actual"].min(),
        data["predicted"].min()
    )

    maximum = max(
        data["actual"].max(),
        data["predicted"].max()
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    ax.set_xlabel(
        "Actual ΔMacro-F1"
    )

    ax.set_ylabel(
        "Predicted ΔMacro-F1"
    )

    ax.set_title(
        "Predicted vs Actual Resampling Effectiveness"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_07_predicted_vs_actual"
    )


def figure_08_statistical_significance():
    df = load_csv(
        STATISTICAL_FRIEDMAN_PATH
    )

    df = df[
        df["analysis"] != "ALL_CLASSIFIERS"
    ].copy()

    df = df.sort_values(
        "p_value",
        ascending=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    y = np.arange(
        len(df)
    )

    values = -np.log10(
        df["p_value"].clip(
            lower=1e-12
        )
    )

    ax.barh(
        y,
        values
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        df["analysis"]
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "-log10(Friedman p-value)"
    )

    ax.set_title(
        "Statistical Significance of Resampling Differences"
    )

    threshold = -np.log10(
        0.05
    )

    ax.axvline(
        threshold,
        linestyle="--",
        label="p = 0.05"
    )

    ax.legend()

    for index, value in enumerate(values):
        ax.text(
            value,
            index,
            f" {df.iloc[index]['p_value']:.4f}",
            va="center"
        )

    fig.tight_layout()

    save_figure(
        fig,
        "figure_08_statistical_significance"
    )


def main():
    print("=" * 75)
    print(
        "ResampleRank - Research Figure Generation"
    )
    print("=" * 75)

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nGenerating figures...\n")

    figure_01_dataset_imbalance()
    figure_02_strategy_macro_f1()
    figure_03_dataset_macro_f1()
    figure_04_meta_model_error()
    figure_05_ranking_accuracy()
    figure_06_computational_savings()
    figure_07_predicted_vs_actual()
    figure_08_statistical_significance()

    print("\n" + "=" * 75)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 75)

    generated = sorted(
        FIGURES_DIR.glob(
            "figure_*.png"
        )
    )

    print(
        f"\nGenerated {len(generated)} PNG figures."
    )

    for path in generated:
        print(path)


if __name__ == "__main__":
    main()