from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
META = RESULTS / "meta_model_lodo_clean43"
POLICY = RESULTS / "selection_baselines_lodo_clean43"
STATS = RESULTS / "statistical_policy_comparison_lodo_clean43"

OUT = ROOT / "figures" / "paper_clean43"
TABLES = OUT / "tables"
OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    "ridge",
    "random_forest",
    "gradient_boosting",
    "xgboost",
]

MODEL_LABELS = {
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
}

CLASSIFIER_ORDER = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
]

CLASSIFIER_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / (name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("FIGURE:", name + ".png and .pdf")


def require_columns(df, columns, filename):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(
            "{} missing required columns: {}".format(filename, missing)
        )


def main():
    overall_path = META / "meta_model_overall.csv"
    policy_path = POLICY / "lodo_policy_summary.csv"
    runtime_path = STATS / "policy_runtime_summary_adjusted_inference.csv"
    robustness_path = STATS / "policy_robustness_by_classifier.csv"
    wilcoxon_path = STATS / "paired_policy_comparisons.csv"
    mean_test_path = STATS / "paired_mean_effect_tests.csv"

    for path in [
        overall_path, policy_path, runtime_path, robustness_path,
        wilcoxon_path, mean_test_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(str(path))

    overall = pd.read_csv(overall_path)
    policy = pd.read_csv(policy_path)
    runtime = pd.read_csv(runtime_path)
    robustness = pd.read_csv(robustness_path)
    wilcoxon = pd.read_csv(wilcoxon_path)
    mean_tests = pd.read_csv(mean_test_path)

    model_col = "model" if "model" in overall.columns else "meta_model"
    require_columns(
        overall,
        [
            model_col, "mae", "rmse",
            "top1_resampling_accuracy",
            "top3_resampling_accuracy",
            "mean_spearman", "mean_kendall", "mean_regret",
        ],
        overall_path.name,
    )
    require_columns(
        policy,
        ["meta_model", "policy", "classifier",
         "mean_improvement_vs_baseline", "mean_regret"],
        policy_path.name,
    )
    require_columns(
        runtime,
        [
            "meta_model", "policy",
            "mean_original_savings_percent",
            "mean_adjusted_savings_percent",
        ],
        runtime_path.name,
    )
    require_columns(
        robustness,
        ["meta_model", "classifier", "comparator", "mean_paired_difference"],
        robustness_path.name,
    )

    overall = overall[
        overall["classifier"].astype(str).str.upper().eq("ALL")
    ].copy()

    overall = overall.set_index(model_col).reindex(MODEL_ORDER).reset_index()
    if len(overall) != 4 or overall[model_col].isna().any():
        raise ValueError("Could not select all four overall meta-model rows.")

    labels = [MODEL_LABELS[x] for x in overall[model_col]]

    # ---------------------------------------------------------------
    # Figure 1: Predictive error
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(overall))
    width = 0.34

    ax.bar(x - width / 2, overall["mae"], width, label="MAE")
    ax.bar(x + width / 2, overall["rmse"], width, label="RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Error in predicted Macro-F1 improvement")
    ax.set_title("LODO meta-model prediction error")
    ax.legend()
    save_figure(fig, "fig01_meta_model_prediction_error")

    # ---------------------------------------------------------------
    # Figure 2: Top-1 and Top-3 strategy selection
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    width = 0.34

    top1 = 100 * overall["top1_resampling_accuracy"]
    top3 = 100 * overall["top3_resampling_accuracy"]

    ax.bar(x - width / 2, top1, width, label="Top-1 accuracy")
    ax.bar(x + width / 2, top3, width, label="Top-3 accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Resampling strategy selection accuracy")
    ax.legend()
    save_figure(fig, "fig02_strategy_selection_accuracy")

    # ---------------------------------------------------------------
    # Figure 3: Ranking correlations
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    width = 0.34

    ax.bar(x - width / 2, overall["mean_spearman"], width, label="Spearman")
    ax.bar(x + width / 2, overall["mean_kendall"], width, label="Kendall")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean rank correlation")
    ax.set_ylim(-0.1, 1.0)
    ax.set_title("Agreement between predicted and actual rankings")
    ax.legend()
    save_figure(fig, "fig03_ranking_correlations")

    # ---------------------------------------------------------------
    # Figure 4: Mean improvement by selection policy
    # ---------------------------------------------------------------
    all_classifiers = policy[
        policy["classifier"].astype(str).str.upper().eq("ALL")
    ].copy()

    if all_classifiers.empty:
        raise ValueError("No overall rows found in the policy summary.")

    global_policies = [
        "always_no_resampling",
        "lodo_training_global_best",
        "oracle_upper_bound",
    ]

    global_rows = (
        all_classifiers[
            all_classifiers["policy"].isin(global_policies)
        ]
        .drop_duplicates("policy")
    )

    selector_rows = all_classifiers[
        all_classifiers["policy"] == "meta_learning_baseline_aware"
    ].copy()

    plot_policy = pd.concat(
        [global_rows, selector_rows],
        ignore_index=True,
    )

    policy_labels = {
        "always_no_resampling": "Always no resampling",
        "lodo_training_global_best": "LODO global best",
        "oracle_upper_bound": "Oracle upper bound",
    }

    plot_policy["plot_label"] = plot_policy.apply(
        lambda row: (
            MODEL_LABELS.get(row["meta_model"], str(row["meta_model"]))
            + " selector"
            if row["policy"] == "meta_learning_baseline_aware"
            else policy_labels.get(row["policy"], row["policy"])
        ),
        axis=1,
    )

    order = [
        "Always no resampling",
        "LODO global best",
        "Ridge selector",
        "Random Forest selector",
        "Gradient Boosting selector",
        "XGBoost selector",
        "Oracle upper bound",
    ]

    plot_policy = plot_policy.set_index("plot_label").reindex(order).dropna(
        subset=["mean_improvement_vs_baseline"]
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 5.7))
    values = plot_policy["mean_improvement_vs_baseline"].to_numpy()
    ypos = np.arange(len(plot_policy))

    ax.barh(ypos, values)
    ax.axvline(0, linewidth=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(plot_policy["plot_label"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean Macro-F1 improvement over no resampling")
    ax.set_title("Selection-policy performance under LODO evaluation")

    save_figure(fig, "fig04_policy_mean_improvement")

    # ---------------------------------------------------------------
    # Figure 5: Adjusted computational savings
    # ---------------------------------------------------------------
    learned = runtime[
        runtime["policy"] == "meta_learning_baseline_aware"
    ].copy()

    learned = learned.set_index("meta_model").reindex(MODEL_ORDER).reset_index()

    if len(learned) != 4 or learned["meta_model"].isna().any():
        raise ValueError("Expected four runtime results for learned selectors.")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    width = 0.34

    ax.bar(
        x - width / 2,
        learned["mean_original_savings_percent"],
        width,
        label="Before inference overhead",
    )
    ax.bar(
        x + width / 2,
        learned["mean_adjusted_savings_percent"],
        width,
        label="After inference overhead",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_LABELS[m] for m in learned["meta_model"]],
        rotation=15,
        ha="right",
    )
    ax.set_ylabel("Estimated runtime savings (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Runtime savings after accounting for inference")
    ax.legend()
    save_figure(fig, "fig05_adjusted_runtime_savings")

    # ---------------------------------------------------------------
    # Figure 6: Classifier-specific robustness versus global best
    # ---------------------------------------------------------------
    robust_global = robustness[
        robustness["comparator"] == "lodo_training_global_best"
    ].copy()

    heat = robust_global.pivot(
        index="meta_model",
        columns="classifier",
        values="mean_paired_difference",
    ).reindex(index=MODEL_ORDER, columns=CLASSIFIER_ORDER)

    if heat.isna().any().any():
        raise ValueError("Incomplete classifier-robustness matrix.")

    matrix = heat.to_numpy()
    limit = max(float(np.abs(matrix).max()), 1e-6)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    image = ax.imshow(
        matrix,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )

    ax.set_xticks(np.arange(len(CLASSIFIER_ORDER)))
    ax.set_xticklabels(
        [CLASSIFIER_LABELS[c] for c in CLASSIFIER_ORDER],
        rotation=20,
        ha="right",
    )
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, "{:.4f}".format(matrix[i, j]),
                ha="center", va="center", fontsize=9,
            )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean paired improvement vs. global best")
    ax.set_title("Classifier-level robustness of meta-learning selection")
    save_figure(fig, "fig06_classifier_robustness")

    # ---------------------------------------------------------------
    # Clean CSV tables for paper preparation
    # ---------------------------------------------------------------
    overall.to_csv(TABLES / "table_meta_model_overall.csv", index=False)

    all_classifiers.to_csv(
        TABLES / "table_policy_summary_overall.csv", index=False
    )

    wilcoxon.to_csv(
        TABLES / "table_paired_wilcoxon_holm.csv", index=False
    )

    mean_tests.to_csv(
        TABLES / "table_paired_mean_effects_holm.csv", index=False
    )

    learned.to_csv(
        TABLES / "table_runtime_savings_adjusted.csv", index=False
    )

    robust_global.to_csv(
        TABLES / "table_classifier_robustness_vs_global.csv", index=False
    )

    print("\n" + "=" * 75)
    print("PAPER FIGURES AND TABLES GENERATED")
    print("=" * 75)
    print("Figure directory:", OUT)
    print("Table directory :", TABLES)
    print("Figures created : 6 PNG files and 6 PDF files")
    print("Tables created  : 6 CSV files")
    print("Source results  : read-only; original result files were not changed")


if __name__ == "__main__":
    main()
