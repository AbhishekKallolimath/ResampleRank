import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = (
    BASE_DIR / "results" / "selection_baselines_lodo_clean43"
    / "lodo_policy_details.csv"
)
OUTPUT_DIR = (
    BASE_DIR / "results" / "statistical_policy_comparison_lodo_clean43"
)

N_BOOTSTRAP = 10000
RANDOM_SEED = 42
TOLERANCE = 1e-12

MODELS = [
    "ridge",
    "random_forest",
    "gradient_boosting",
    "xgboost",
]
COMPARATORS = [
    "always_no_resampling",
    "lodo_training_global_best",
]
SELECTOR = "meta_learning_baseline_aware"
KEYS = ["meta_model", "dataset", "classifier", "policy"]


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    total = len(p_values)

    for rank, index in enumerate(order):
        candidate = (total - rank) * p_values[index]
        running_max = max(running_max, candidate)
        adjusted[index] = min(1.0, running_max)

    return adjusted


def paired_test(differences, rng):
    differences = np.asarray(differences, dtype=float)
    n = len(differences)

    mean_difference = float(np.mean(differences))
    median_difference = float(np.median(differences))

    bootstrap_means = rng.choice(
        differences,
        size=(N_BOOTSTRAP, n),
        replace=True,
    ).mean(axis=1)

    ci_low, ci_high = np.percentile(
        bootstrap_means, [2.5, 97.5]
    )

    wins = int(np.sum(differences > TOLERANCE))
    losses = int(np.sum(differences < -TOLERANCE))
    ties = int(n - wins - losses)

    if np.all(np.abs(differences) <= TOLERANCE):
        statistic, p_value = 0.0, 1.0
    else:
        result = wilcoxon(
            differences,
            zero_method="wilcox",
            alternative="two-sided",
            mode="auto",
        )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)

    return {
        "datasets": n,
        "mean_difference": mean_difference,
        "median_difference": median_difference,
        "bootstrap_95ci_lower": float(ci_low),
        "bootstrap_95ci_upper": float(ci_high),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(str(INPUT_FILE))

    df = pd.read_csv(INPUT_FILE)

    required = KEYS + ["selected_improvement_vs_baseline"]
    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError("Missing columns: {}".format(missing_columns))

    if len(df) != 2752:
        raise ValueError(
            "Expected 2752 rows; found {}".format(len(df))
        )

    if df[KEYS].isna().any().any():
        raise ValueError("Grouping keys contain missing values.")

    if df.duplicated(KEYS).any():
        raise ValueError("Duplicate model/dataset/classifier/policy rows.")

    observed_models = set(df["meta_model"].unique())
    if observed_models != set(MODELS):
        raise ValueError("Unexpected meta-model names: {}".format(
            sorted(observed_models)
        ))

    classifier_counts = df.groupby(
        ["meta_model", "dataset", "policy"]
    )["classifier"].nunique()

    if not (classifier_counts == 4).all():
        raise ValueError(
            "Expected four base classifiers in every model/dataset/policy."
        )

    # Average the four classifier results within each dataset.
    dataset_values = (
        df.groupby(
            ["meta_model", "dataset", "policy"],
            as_index=False,
        )
        .agg(
            mean_improvement=(
                "selected_improvement_vs_baseline", "mean"
            ),
            classifier_count=("classifier", "nunique"),
        )
    )

    pivot = dataset_values.pivot(
        index=["meta_model", "dataset"],
        columns="policy",
        values="mean_improvement",
    )
    pivot.columns.name = None

    required_policies = [SELECTOR] + COMPARATORS
    if not set(required_policies).issubset(pivot.columns):
        raise ValueError("One or more comparison policies are missing.")

    if pivot[required_policies].isna().any().any():
        raise ValueError("Missing values found after pairing policies.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_values.to_csv(
        OUTPUT_DIR / "dataset_level_policy_values.csv",
        index=False,
    )

    rng = np.random.RandomState(RANDOM_SEED)
    rows = []

    for model in MODELS:
        model_values = pivot.loc[model]

        if len(model_values) != 43:
            raise ValueError(
                "{}: expected 43 datasets, got {}".format(
                    model, len(model_values)
                )
            )

        for comparator in COMPARATORS:
            # Positive means the meta-learning selector performs better.
            differences = (
                model_values[SELECTOR].to_numpy(dtype=float)
                - model_values[comparator].to_numpy(dtype=float)
            )

            result = paired_test(differences, rng)
            result["meta_model"] = model
            result["comparison"] = (
                SELECTOR + "_vs_" + comparator
            )
            rows.append(result)

    results = pd.DataFrame(rows)
    results["holm_adjusted_p_value"] = holm_adjust(
        results["p_value"].to_numpy()
    )

    ordered_columns = [
        "meta_model",
        "comparison",
        "datasets",
        "mean_difference",
        "median_difference",
        "bootstrap_95ci_lower",
        "bootstrap_95ci_upper",
        "wins",
        "ties",
        "losses",
        "wilcoxon_statistic",
        "p_value",
        "holm_adjusted_p_value",
    ]
    results = results[ordered_columns]

    results.to_csv(
        OUTPUT_DIR / "paired_policy_comparisons.csv",
        index=False,
    )

    notes = (
        "Analysis unit: dataset. Four classifier outcomes are averaged "
        "within each dataset before pairing policies.\n"
        "Effect: mean selected Macro-F1 improvement difference; positive "
        "values favor the meta-learning selector.\n"
        "Intervals: percentile paired-dataset bootstrap, 10000 resamples, "
        "seed 42.\n"
        "Tests: two-sided Wilcoxon signed-rank tests; Holm correction "
        "across eight comparisons.\n"
        "Caution: LODO training folds overlap. Treat p-values and confidence "
        "intervals as exploratory rather than definitive independent-fold "
        "inference. Runtime savings are not evaluated in this analysis.\n"
    )
    (OUTPUT_DIR / "method_notes.txt").write_text(
        notes, encoding="utf-8"
    )

    print("=" * 78)
    print("PAIRED LODO POLICY COMPARISONS")
    print("=" * 78)
    print(results.to_string(index=False, float_format=lambda x: "{:.6f}".format(x)))
    print("\nSaved output directory:", OUTPUT_DIR)
    print("Comparisons:", len(results))
    print("All comparisons use 43 paired dataset-level observations.")


if __name__ == "__main__":
    main()
