from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare


RESULTS_DIR = Path("results")
OUTPUT_DIR = RESULTS_DIR / "statistical_analysis"

PRIMARY_DATASETS = [
    "pima",
    "glass1",
    "wisconsin",
    "yeast1",
    "haberman",
    "vehicle2",
    "iris0",
    "segment0",
    "page-blocks0",
    "vowel0",
    "shuttle-c0-vs-c4",
    "ecoli-0_vs_1",
]

CLASSIFIERS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
]

STRATEGIES = [
    "No Resampling",
    "Random Oversampling",
    "Random Undersampling",
    "SMOTE",
    "ADASYN",
    "SMOTE-Tomek",
    "SMOTE-ENN",
]

STRATEGY_MAP = {
    "none": "No Resampling",
    "no_resampling": "No Resampling",
    "no resampling": "No Resampling",
    "No Resampling": "No Resampling",

    "ros": "Random Oversampling",
    "random_oversampling": "Random Oversampling",
    "random oversampling": "Random Oversampling",
    "Random Oversampling": "Random Oversampling",

    "rus": "Random Undersampling",
    "random_undersampling": "Random Undersampling",
    "random undersampling": "Random Undersampling",
    "Random Undersampling": "Random Undersampling",

    "smote": "SMOTE",
    "SMOTE": "SMOTE",

    "adasyn": "ADASYN",
    "ADASYN": "ADASYN",

    "smote_tomek": "SMOTE-Tomek",
    "smote-tomek": "SMOTE-Tomek",
    "SMOTE-Tomek": "SMOTE-Tomek",

    "smote_enn": "SMOTE-ENN",
    "smote-enn": "SMOTE-ENN",
    "SMOTE-ENN": "SMOTE-ENN",
}


def normalize_strategy(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value in STRATEGY_MAP:
        return STRATEGY_MAP[value]

    key = value.lower().replace(" ", "_")

    if key in STRATEGY_MAP:
        return STRATEGY_MAP[key]

    return value


def find_metric_column(df):
    candidates = [
        "macro_f1_code",
        "macro_f1",
        "Macro-F1",
        "macro_f1_score",
        "f1_macro",
        "target_macro_f1",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    normalized = {
        str(column).lower().replace("-", "_"): column
        for column in df.columns
    }

    for column in [
        "macro_f1_code",
        "macro_f1",
        "macro_f1_score",
        "f1_macro",
        "target_macro_f1",
    ]:
        if column in normalized:
            return normalized[column]

    raise ValueError(
        "Could not find a Macro-F1 column. "
        f"Available columns: {list(df.columns)}"
    )


def load_results():
    records = []

    for dataset in PRIMARY_DATASETS:
        for classifier in CLASSIFIERS:
            filename = f"{dataset}_{classifier}_results.csv"
            path = RESULTS_DIR / filename

            if not path.exists():
                print(f"WARNING: Missing file: {path}")
                continue

            df = pd.read_csv(path)

            metric_column = find_metric_column(df)

            strategy_column = None

            for candidate in [
                "resampling",
                "resampling_code",
            ]:
                if candidate in df.columns:
                    strategy_column = candidate
                    break

            if strategy_column is None:
                raise ValueError(
                    f"No resampling column found in {path}"
                )

            temp = pd.DataFrame({
                "dataset": dataset,
                "classifier": classifier,
                "resampling": (
                    df[strategy_column]
                    .map(normalize_strategy)
                ),
                "macro_f1": pd.to_numeric(
                    df[metric_column],
                    errors="coerce"
                ),
            })

            temp = temp[
                temp["resampling"].isin(STRATEGIES)
            ]

            temp = temp.dropna(
                subset=["macro_f1"]
            )

            records.append(temp)

            print(
                f"Loaded {filename} -> "
                f"{len(temp)} strategy rows"
            )

    if not records:
        raise RuntimeError(
            "No benchmark result files were found."
        )

    return pd.concat(
        records,
        ignore_index=True
    )


def make_pivot(
    df,
    classifier=None
):
    temp = df.copy()

    if classifier is not None:
        temp = temp[
            temp["classifier"] == classifier
        ]

    temp = (
        temp.groupby(
            ["dataset", "resampling"],
            as_index=False
        )["macro_f1"]
        .mean()
    )

    pivot = temp.pivot(
        index="dataset",
        columns="resampling",
        values="macro_f1"
    )

    for strategy in STRATEGIES:
        if strategy not in pivot.columns:
            pivot[strategy] = np.nan

    pivot = pivot[STRATEGIES]

    pivot = pivot.dropna(
        subset=STRATEGIES,
        how="any"
    )

    return pivot


def friedman_analysis(
    pivot,
    analysis_name
):
    arrays = [
        pivot[strategy].values
        for strategy in STRATEGIES
    ]

    if len(pivot) < 2:
        return {
            "analysis": analysis_name,
            "datasets": len(pivot),
            "strategies": len(STRATEGIES),
            "friedman_statistic": np.nan,
            "p_value": np.nan,
            "kendall_w": np.nan,
        }

    statistic, p_value = friedmanchisquare(
        *arrays
    )

    n = len(pivot)
    k = len(STRATEGIES)

    kendall_w = (
        statistic /
        (n * (k - 1))
    )

    return {
        "analysis": analysis_name,
        "datasets": n,
        "strategies": k,
        "friedman_statistic": statistic,
        "p_value": p_value,
        "kendall_w": kendall_w,
    }


def rank_biserial_effect(
    x,
    y
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    diff = x - y

    diff = diff[
        np.isfinite(diff)
    ]

    diff = diff[
        diff != 0
    ]

    if len(diff) == 0:
        return 0.0

    abs_diff = np.abs(diff)

    order = np.argsort(abs_diff)

    ranks = np.empty(
        len(diff),
        dtype=float
    )

    sorted_abs = abs_diff[order]

    i = 0

    while i < len(sorted_abs):
        j = i

        while (
            j + 1 < len(sorted_abs)
            and sorted_abs[j + 1]
            == sorted_abs[i]
        ):
            j += 1

        average_rank = (
            i + j + 2
        ) / 2.0

        ranks[
            order[i:j + 1]
        ] = average_rank

        i = j + 1

    positive_rank_sum = ranks[
        diff > 0
    ].sum()

    negative_rank_sum = ranks[
        diff < 0
    ].sum()

    denominator = (
        len(diff) *
        (len(diff) + 1)
    ) / 2.0

    if denominator == 0:
        return 0.0

    return (
        positive_rank_sum -
        negative_rank_sum
    ) / denominator


def holm_adjust(
    p_values
):
    p_values = np.asarray(
        p_values,
        dtype=float
    )

    adjusted = np.full(
        len(p_values),
        np.nan,
        dtype=float
    )

    valid = np.isfinite(
        p_values
    )

    if not np.any(valid):
        return adjusted

    indices = np.where(valid)[0]

    sorted_indices = indices[
        np.argsort(
            p_values[valid]
        )
    ]

    m = len(sorted_indices)

    running_max = 0.0

    for rank, idx in enumerate(
        sorted_indices
    ):
        adjusted_value = (
            m - rank
        ) * p_values[idx]

        running_max = max(
            running_max,
            adjusted_value
        )

        adjusted[idx] = min(
            running_max,
            1.0
        )

    return adjusted


def exact_paired_permutation_test(
    x,
    y
):
    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    diff = x - y

    diff = diff[
        np.isfinite(diff)
    ]

    diff = diff[
        diff != 0
    ]

    if len(diff) == 0:
        return np.nan, np.nan

    observed = np.sum(diff)

    n = len(diff)

    total = 2 ** n
    extreme = 0

    for mask in range(total):
        signs = np.ones(n)

        for i in range(n):
            if mask & (1 << i):
                signs[i] = -1.0

        statistic = np.sum(
            signs * diff
        )

        if (
            abs(statistic)
            >= abs(observed)
        ):
            extreme += 1

    p_value = extreme / total

    return observed, p_value


def pairwise_analysis(
    pivot,
    analysis_name
):
    rows = []

    for strategy_a, strategy_b in itertools.combinations(
        STRATEGIES,
        2
    ):
        x = pivot[strategy_a].values
        y = pivot[strategy_b].values

        permutation_statistic, p_value = (
            exact_paired_permutation_test(
                x,
                y
            )
        )

        effect = rank_biserial_effect(
            x,
            y
        )

        mean_a = float(
            np.mean(x)
        )

        mean_b = float(
            np.mean(y)
        )

        mean_difference = (
            mean_a - mean_b
        )

        rows.append({
            "analysis": analysis_name,
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
            "datasets": len(x),
            "mean_a": mean_a,
            "mean_b": mean_b,
            "mean_difference_a_minus_b": (
                mean_difference
            ),
            "permutation_statistic": (
                permutation_statistic
            ),
            "raw_p_value": p_value,
            "rank_biserial_effect": effect,
        })

    result = pd.DataFrame(
        rows
    )

    result[
        "holm_adjusted_p_value"
    ] = holm_adjust(
        result["raw_p_value"].values
    )

    result[
        "significant_alpha_0_05"
    ] = (
        result[
            "holm_adjusted_p_value"
        ] < 0.05
    )

    return result


def summary_analysis(
    pivot,
    analysis_name
):
    baseline = pivot[
        "No Resampling"
    ]

    rows = []

    for strategy in STRATEGIES:
        values = pivot[strategy]

        differences = (
            values - baseline
        )

        wins = int(
            np.sum(
                differences > 1e-12
            )
        )

        ties = int(
            np.sum(
                np.abs(differences)
                <= 1e-12
            )
        )

        losses = int(
            np.sum(
                differences < -1e-12
            )
        )

        rows.append({
            "analysis": analysis_name,
            "strategy": strategy,
            "datasets": len(values),
            "mean_macro_f1": (
                values.mean()
            ),
            "median_macro_f1": (
                values.median()
            ),
            "std_macro_f1": (
                values.std(ddof=1)
            ),
            "mean_improvement_vs_baseline": (
                differences.mean()
            ),
            "median_improvement_vs_baseline": (
                differences.median()
            ),
            "wins_vs_baseline": wins,
            "ties_vs_baseline": ties,
            "losses_vs_baseline": losses,
            "improvement_rate_percent": (
                100.0 *
                wins /
                len(values)
            ),
        })

    return pd.DataFrame(
        rows
    )


def main():
    print("=" * 75)
    print(
        "ResampleRank - Statistical Significance Analysis"
    )
    print("=" * 75)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "\nLoading benchmark results...\n"
    )

    data = load_results()

    print(
        "\nTotal loaded rows:",
        len(data)
    )

    all_friedman = []
    all_posthoc = []
    all_summary = []

    for classifier in CLASSIFIERS:
        print(
            "\n" + "-" * 75
        )

        print(
            f"Classifier: {classifier}"
        )

        print(
            "-" * 75
        )

        pivot = make_pivot(
            data,
            classifier=classifier
        )

        print(
            f"Complete datasets: "
            f"{len(pivot)}"
        )

        if len(pivot) == 0:
            continue

        friedman_result = (
            friedman_analysis(
                pivot,
                classifier
            )
        )

        all_friedman.append(
            friedman_result
        )

        print(
            f"Friedman statistic: "
            f"{friedman_result['friedman_statistic']:.6f}"
        )

        print(
            f"Friedman p-value: "
            f"{friedman_result['p_value']:.6f}"
        )

        print(
            f"Kendall's W: "
            f"{friedman_result['kendall_w']:.6f}"
        )

        posthoc = pairwise_analysis(
            pivot,
            classifier
        )

        all_posthoc.append(
            posthoc
        )

        summary = summary_analysis(
            pivot,
            classifier
        )

        all_summary.append(
            summary
        )

    print(
        "\n" + "-" * 75
    )

    print(
        "Overall analysis across classifiers"
    )

    print(
        "-" * 75
    )

    pooled = (
        data.groupby(
            ["dataset", "resampling"],
            as_index=False
        )["macro_f1"]
        .mean()
    )

    pooled_pivot = pooled.pivot(
        index="dataset",
        columns="resampling",
        values="macro_f1"
    )

    for strategy in STRATEGIES:
        if strategy not in pooled_pivot.columns:
            pooled_pivot[strategy] = np.nan

    pooled_pivot = pooled_pivot[
        STRATEGIES
    ]

    pooled_pivot = pooled_pivot.dropna(
        subset=STRATEGIES,
        how="any"
    )

    print(
        f"Complete pooled datasets: "
        f"{len(pooled_pivot)}"
    )

    if len(pooled_pivot) > 0:

        pooled_friedman = (
            friedman_analysis(
                pooled_pivot,
                "ALL_CLASSIFIERS"
            )
        )

        all_friedman.append(
            pooled_friedman
        )

        print(
            f"Friedman statistic: "
            f"{pooled_friedman['friedman_statistic']:.6f}"
        )

        print(
            f"Friedman p-value: "
            f"{pooled_friedman['p_value']:.6f}"
        )

        print(
            f"Kendall's W: "
            f"{pooled_friedman['kendall_w']:.6f}"
        )

        pooled_posthoc = (
            pairwise_analysis(
                pooled_pivot,
                "ALL_CLASSIFIERS"
            )
        )

        all_posthoc.append(
            pooled_posthoc
        )

        pooled_summary = (
            summary_analysis(
                pooled_pivot,
                "ALL_CLASSIFIERS"
            )
        )

        all_summary.append(
            pooled_summary
        )

    friedman_df = pd.DataFrame(
        all_friedman
    )

    posthoc_df = pd.concat(
        all_posthoc,
        ignore_index=True
    )

    summary_df = pd.concat(
        all_summary,
        ignore_index=True
    )

    effect_df = posthoc_df[
        [
            "analysis",
            "strategy_a",
            "strategy_b",
            "datasets",
            "rank_biserial_effect",
            "holm_adjusted_p_value",
            "significant_alpha_0_05",
        ]
    ].copy()

    effect_df[
        "comparison"
    ] = (
        effect_df[
            "strategy_a"
        ]
        + " vs "
        + effect_df[
            "strategy_b"
        ]
    )

    effect_df[
        "interpretation"
    ] = np.where(
        effect_df[
            "rank_biserial_effect"
        ] > 0,
        "positive",
        np.where(
            effect_df[
                "rank_biserial_effect"
            ] < 0,
            "negative",
            "zero"
        )
    )

    effect_df = effect_df[
        [
            "analysis",
            "comparison",
            "datasets",
            "rank_biserial_effect",
            "interpretation",
            "holm_adjusted_p_value",
            "significant_alpha_0_05",
        ]
    ]

    friedman_path = (
        OUTPUT_DIR /
        "friedman_results.csv"
    )

    posthoc_path = (
        OUTPUT_DIR /
        "posthoc_results.csv"
    )

    effect_path = (
        OUTPUT_DIR /
        "effect_sizes.csv"
    )

    summary_path = (
        OUTPUT_DIR /
        "statistical_summary.csv"
    )

    friedman_df.to_csv(
        friedman_path,
        index=False
    )

    posthoc_df.to_csv(
        posthoc_path,
        index=False
    )

    effect_df.to_csv(
        effect_path,
        index=False
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    print(
        "\n" + "=" * 75
    )

    print(
        "STATISTICAL ANALYSIS SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        friedman_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved files:"
    )

    print(
        friedman_path.resolve()
    )

    print(
        posthoc_path.resolve()
    )

    print(
        effect_path.resolve()
    )

    print(
        summary_path.resolve()
    )


if __name__ == "__main__":
    main()