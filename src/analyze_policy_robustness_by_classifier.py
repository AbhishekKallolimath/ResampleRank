import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
INPUT = BASE / "results/selection_baselines_lodo_clean43/lodo_policy_details.csv"
OUT_DIR = BASE / "results/statistical_policy_comparison_lodo_clean43"
OUTPUT = OUT_DIR / "policy_robustness_by_classifier.csv"

MODELS = ["ridge", "random_forest", "gradient_boosting", "xgboost"]
CLASSIFIERS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
]
SELECTOR = "meta_learning_baseline_aware"
COMPARATORS = [
    "always_no_resampling",
    "lodo_training_global_best",
]
N_BOOTSTRAP = 10000
SEED = 42
TOL = 1e-12


def main():
    df = pd.read_csv(INPUT)

    keys = ["meta_model", "dataset", "classifier", "policy"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate model/dataset/classifier/policy rows.")

    needed = [
        "selected_improvement_vs_baseline",
        "meta_model", "dataset", "classifier", "policy",
    ]
    if any(column not in df.columns for column in needed):
        raise ValueError("An expected input column is missing.")

    relevant = df[df["policy"].isin([SELECTOR] + COMPARATORS)]
    paired = relevant.pivot(
        index=["meta_model", "dataset", "classifier"],
        columns="policy",
        values="selected_improvement_vs_baseline",
    ).reset_index()

    rng = np.random.RandomState(SEED)
    rows = []

    for model in MODELS:
        for classifier in CLASSIFIERS:
            subset = paired[
                (paired["meta_model"] == model)
                & (paired["classifier"] == classifier)
            ].sort_values("dataset")

            if len(subset) != 43:
                raise ValueError(
                    "{} / {}: expected 43 datasets, found {}".format(
                        model, classifier, len(subset)
                    )
                )

            for comparator in COMPARATORS:
                differences = (
                    subset[SELECTOR].to_numpy(dtype=float)
                    - subset[comparator].to_numpy(dtype=float)
                )
                boot = rng.choice(
                    differences,
                    size=(N_BOOTSTRAP, len(differences)),
                    replace=True,
                ).mean(axis=1)
                low, high = np.percentile(boot, [2.5, 97.5])

                rows.append({
                    "meta_model": model,
                    "classifier": classifier,
                    "comparator": comparator,
                    "datasets": len(differences),
                    "mean_selector_improvement": float(
                        subset[SELECTOR].mean()
                    ),
                    "mean_comparator_improvement": float(
                        subset[comparator].mean()
                    ),
                    "mean_paired_difference": float(
                        differences.mean()
                    ),
                    "median_paired_difference": float(
                        np.median(differences)
                    ),
                    "bootstrap_95ci_lower": float(low),
                    "bootstrap_95ci_upper": float(high),
                    "wins": int((differences > TOL).sum()),
                    "ties": int((np.abs(differences) <= TOL).sum()),
                    "losses": int((differences < -TOL).sum()),
                })

    result = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)

    print("=" * 90)
    print("POLICY ROBUSTNESS BY BASE CLASSIFIER")
    print("=" * 90)
    print(result.to_string(
        index=False,
        float_format=lambda value: "{:.6f}".format(value),
    ))
    print("\nRows:", len(result))
    print("Expected rows: 32")
    print("Saved:", OUTPUT)
    print(
        "\nPositive paired differences favor the meta-learning selector. "
        "Bootstrap intervals are descriptive; LODO fold dependence remains."
    )


if __name__ == "__main__":
    main()
