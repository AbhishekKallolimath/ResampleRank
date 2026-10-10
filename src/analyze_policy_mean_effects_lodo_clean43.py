import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import t, ttest_1samp

BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / "results" / "selection_baselines_lodo_clean43" / "lodo_policy_details.csv"
OUTPUT = BASE / "results" / "statistical_policy_comparison_lodo_clean43" / "paired_mean_effect_tests.csv"

MODELS = ["ridge", "random_forest", "gradient_boosting", "xgboost"]
SELECTOR = "meta_learning_baseline_aware"
COMPARATORS = ["always_no_resampling", "lodo_training_global_best"]


def holm_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running_max = 0.0
    m = len(p_values)

    for rank, index in enumerate(order):
        running_max = max(running_max, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running_max)

    return adjusted


def main():
    df = pd.read_csv(INPUT)

    # Average base-classifier outcomes within each dataset.
    dataset_values = (
        df.groupby(["meta_model", "dataset", "policy"], as_index=False)
        ["selected_improvement_vs_baseline"].mean()
    )

    paired = dataset_values.pivot(
        index=["meta_model", "dataset"],
        columns="policy",
        values="selected_improvement_vs_baseline",
    )

    rows = []

    for model in MODELS:
        model_data = paired.loc[model]

        if len(model_data) != 43:
            raise ValueError(
                "{}: expected 43 datasets, found {}".format(
                    model, len(model_data)
                )
            )

        for comparator in COMPARATORS:
            differences = (
                model_data[SELECTOR] - model_data[comparator]
            ).to_numpy(dtype=float)

            n = len(differences)
            mean_diff = float(np.mean(differences))
            sd = float(np.std(differences, ddof=1))
            se = sd / np.sqrt(n)
            critical = float(t.ppf(0.975, df=n - 1))

            test = ttest_1samp(differences, popmean=0.0)

            rows.append({
                "meta_model": model,
                "comparison": SELECTOR + "_vs_" + comparator,
                "datasets": n,
                "mean_difference": mean_diff,
                "standard_deviation": sd,
                "standard_error": se,
                "ci_95_lower": mean_diff - critical * se,
                "ci_95_upper": mean_diff + critical * se,
                "t_statistic": float(test.statistic),
                "degrees_of_freedom": n - 1,
                "p_value": float(test.pvalue),
                "positive_datasets": int(np.sum(differences > 1e-12)),
                "tie_datasets": int(np.sum(np.abs(differences) <= 1e-12)),
                "negative_datasets": int(np.sum(differences < -1e-12)),
            })

    results = pd.DataFrame(rows)
    results["holm_adjusted_p_value"] = holm_adjust(
        results["p_value"].to_numpy()
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)

    print("=" * 100)
    print("PAIRED MEAN-EFFECT TESTS")
    print("=" * 100)
    print(results.to_string(
        index=False,
        float_format=lambda value: "{:.6f}".format(value)
    ))
    print("\nSaved:", OUTPUT)
    print("\nPositive mean differences favor the meta-learning selector.")
    print("Confidence intervals and t-tests target the mean paired difference.")
    print("Interpret p-values cautiously because LODO training folds overlap.")


if __name__ == "__main__":
    main()
