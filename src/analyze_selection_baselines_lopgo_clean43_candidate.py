from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
META_FILE = RESULTS_DIR / "meta_dataset_clean43_parent_groups_candidate.csv"
LOPGO_DIR = RESULTS_DIR / "meta_model_lopgo_clean43_candidate"
OUTPUT_DIR = RESULTS_DIR / "selection_baselines_lopgo_clean43_candidate"

MODELS = [
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

KEYS = ["dataset", "classifier", "resampling_code"]

POLICIES = [
    "always_no_resampling",
    "lopgo_training_global_best",
    "meta_learning_baseline_aware",
    "oracle_upper_bound",
]


def safe_correlations(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if (
        len(actual) < 2
        or len(np.unique(actual)) < 2
        or len(np.unique(predicted)) < 2
    ):
        return np.nan, np.nan

    sp = spearmanr(actual, predicted)[0]
    kt = kendalltau(actual, predicted)[0]

    return (
        float(sp) if np.isfinite(sp) else np.nan,
        float(kt) if np.isfinite(kt) else np.nan,
    )


def load_meta():
    if not META_FILE.exists():
        raise FileNotFoundError(META_FILE)

    df = pd.read_csv(META_FILE)

    required = KEYS + [
        "split",
        "resampling",
        "target_macro_f1",
        "baseline_macro_f1",
        "delta_macro_f1_vs_baseline",
        "execution_time_seconds",
        "parent_group",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Meta-dataset is missing columns: {}".format(missing)
        )

    if len(df) != 1204 or df["dataset"].nunique() != 43:
        raise ValueError(
            "Expected 1,204 rows and 43 datasets."
        )

    if df.duplicated(KEYS).any():
        raise ValueError("Duplicate experiment keys found.")

    if not df.groupby("dataset").size().eq(28).all():
        raise ValueError("Each dataset must have 28 experiment rows.")

    if not np.isfinite(
        df[
            [
                "target_macro_f1",
                "baseline_macro_f1",
                "delta_macro_f1_vs_baseline",
                "execution_time_seconds",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite actual values or runtimes found.")

    return df


def load_oof_predictions(meta, model):
    frames = []

    for classifier in CLASSIFIERS:
        path = LOPGO_DIR / "{}_{}_predictions.csv".format(
            model, classifier
        )

        if not path.exists():
            raise FileNotFoundError(path)

        pred = pd.read_csv(path)

        required = KEYS + [
            "predicted_delta_macro_f1",
            "predicted_macro_f1",
        ]

        missing = [c for c in required if c not in pred.columns]
        if missing:
            raise ValueError(
                "{} is missing columns: {}".format(path.name, missing)
            )

        if pred.duplicated(KEYS).any():
            raise ValueError(
                "Duplicate prediction keys in {}".format(path.name)
            )

        frames.append(pred[required])

    pred = pd.concat(frames, ignore_index=True)

    if len(pred) != 1204 or pred.duplicated(KEYS).any():
        raise ValueError(
            "{}: expected 1,204 unique out-of-fold predictions.".format(
                model
            )
        )

    combined = meta.merge(
        pred,
        on=KEYS,
        how="left",
        validate="one_to_one",
    )

    prediction_columns = [
        "predicted_delta_macro_f1",
        "predicted_macro_f1",
    ]

    if combined[prediction_columns].isna().any().any():
        raise ValueError(
            "{}: missing out-of-fold predictions after merge.".format(
                model
            )
        )

    if not np.isfinite(
        combined[prediction_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "{}: non-finite out-of-fold predictions.".format(model)
        )

    return combined


def choose_lopgo_global_strategy(meta, parent_group, classifier):
    train = meta[
        (meta["parent_group"] != parent_group)
        & (meta["classifier"] == classifier)
    ]

    if train["parent_group"].nunique() != meta["parent_group"].nunique() - 1:
        raise ValueError(
            "Expected all other parent groups for {} / {}.".format(
                parent_group, classifier
            )
        )

    means = (
        train.groupby("resampling_code")["delta_macro_f1_vs_baseline"]
        .mean()
        .rename("training_mean_delta")
        .reset_index()
    )

    means = means.sort_values(
        ["training_mean_delta", "resampling_code"],
        ascending=[False, True],
        kind="mergesort",
    )

    best = means.iloc[0]
    return best["resampling_code"], float(best["training_mean_delta"])


def calculate_ranking_metrics(group, model):
    actual = group["target_macro_f1"].to_numpy(dtype=float)
    predicted = group["predicted_delta_macro_f1"].to_numpy(dtype=float)

    best_actual = float(actual.max())
    best_codes = set(
        group.loc[
            np.isclose(actual, best_actual, rtol=0.0, atol=1e-12),
            "resampling_code",
        ]
    )

    ranked = group.sort_values(
        ["predicted_delta_macro_f1", "resampling_code"],
        ascending=[False, True],
        kind="mergesort",
    )

    predicted_best = ranked.iloc[0]["resampling_code"]
    top3_codes = set(ranked.head(3)["resampling_code"])

    sp, kt = safe_correlations(actual, predicted)

    return {
        "meta_model": model,
        "dataset": group.iloc[0]["dataset"],
        "classifier": group.iloc[0]["classifier"],
        "actual_best_codes": "|".join(sorted(best_codes)),
        "predicted_best_code": predicted_best,
        "top1_correct": predicted_best in best_codes,
        "top3_correct": bool(top3_codes.intersection(best_codes)),
        "spearman": sp,
        "kendall": kt,
    }


def evaluate_policies(meta, predictions, model):
    details = []
    ranking_rows = []
    global_rows = []

    for (dataset, classifier), group in predictions.groupby(
        ["dataset", "classifier"],
        sort=True,
    ):
        group = group.copy()

        if len(group) != 7 or group["resampling_code"].nunique() != 7:
            raise ValueError(
                "{} / {} must contain seven strategies.".format(
                    dataset, classifier
                )
            )

        baseline_rows = group[group["resampling_code"] == "none"]
        if len(baseline_rows) != 1:
            raise ValueError(
                "{} / {} needs one no-resampling baseline.".format(
                    dataset, classifier
                )
            )

        baseline = float(baseline_rows.iloc[0]["baseline_macro_f1"])

        if not np.allclose(
            group["baseline_macro_f1"].to_numpy(dtype=float),
            baseline,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(
                "Inconsistent baseline scores for {} / {}.".format(
                    dataset, classifier
                )
            )

        actual_best = float(group["target_macro_f1"].max())
        best_codes = set(
            group.loc[
                np.isclose(
                    group["target_macro_f1"],
                    actual_best,
                    rtol=0.0,
                    atol=1e-12,
                ),
                "resampling_code",
            ]
        )

        full_runtime = float(group["execution_time_seconds"].sum())
        if full_runtime <= 0:
            raise ValueError("Non-positive evaluation runtime.")

        positive = group[
            (group["resampling_code"] != "none")
            & (group["predicted_delta_macro_f1"] > 0)
        ]

        if positive.empty:
            learned_code = "none"
        else:
            learned_code = (
                positive.sort_values(
                    ["predicted_delta_macro_f1", "resampling_code"],
                    ascending=[False, True],
                    kind="mergesort",
                )
                .iloc[0]["resampling_code"]
            )

        parent_group = group.iloc[0]["parent_group"]
        global_code, global_delta = choose_lopgo_global_strategy(
            meta, parent_group, classifier
        )

        raw_ranking = calculate_ranking_metrics(group, model)
        ranking_rows.append(raw_ranking)

        train_dataset_count = meta.loc[
            meta["parent_group"] != parent_group, "dataset"
        ].nunique()

        global_rows.append({
            "held_out_dataset": dataset,
            "held_out_parent_group": parent_group,
            "classifier": classifier,
            "training_datasets_used": train_dataset_count,
            "selected_global_strategy": global_code,
            "training_mean_delta": global_delta,
        })

        oracle_code = sorted(best_codes)[0]

        selected_codes = {
            "always_no_resampling": "none",
            "lopgo_training_global_best": global_code,
            "meta_learning_baseline_aware": learned_code,
            "oracle_upper_bound": oracle_code,
        }

        for policy, code in selected_codes.items():
            selected = group[group["resampling_code"] == code]

            if len(selected) != 1:
                raise ValueError(
                    "Strategy {} not found uniquely for {} / {}.".format(
                        code, dataset, classifier
                    )
                )

            row = selected.iloc[0]
            score = float(row["target_macro_f1"])
            runtime = float(row["execution_time_seconds"])

            details.append({
                "meta_model": model,
                "dataset": dataset,
                "classifier": classifier,
                "policy": policy,
                "selected_resampling_code": code,
                "selected_resampling": row["resampling"],
                "baseline_macro_f1": baseline,
                "selected_macro_f1": score,
                "selected_improvement_vs_baseline": score - baseline,
                "oracle_best_macro_f1": actual_best,
                "oracle_improvement_vs_baseline": actual_best - baseline,
                "regret": actual_best - score,
                "matches_oracle": code in best_codes,
                "used_resampling": code != "none",
                "selected_runtime_seconds": runtime,
                "full_seven_strategy_runtime_seconds": full_runtime,
                "estimated_runtime_savings_seconds": full_runtime - runtime,
                "estimated_runtime_savings_percent": (
                    (full_runtime - runtime) / full_runtime * 100.0
                ),
            })

    return (
        pd.DataFrame(details),
        pd.DataFrame(ranking_rows),
        pd.DataFrame(global_rows),
    )


def summarize(details, ranking):
    rows = []

    for (model, policy), policy_df in details.groupby(
        ["meta_model", "policy"],
        sort=True,
    ):
        classifier_groups = [
            (c, policy_df[policy_df["classifier"] == c])
            for c in CLASSIFIERS
        ]
        classifier_groups.append(("ALL", policy_df))

        for classifier, group in classifier_groups:
            if group.empty:
                continue

            rows.append({
                "meta_model": model,
                "policy": policy,
                "classifier": classifier,
                "dataset_classifier_cases": len(group),
                "datasets": group["dataset"].nunique(),
                "mean_selected_macro_f1": group["selected_macro_f1"].mean(),
                "mean_improvement_vs_baseline": (
                    group["selected_improvement_vs_baseline"].mean()
                ),
                "mean_regret": group["regret"].mean(),
                "oracle_match_rate_percent": (
                    group["matches_oracle"].mean() * 100.0
                ),
                "resampling_selection_rate_percent": (
                    group["used_resampling"].mean() * 100.0
                ),
                "mean_estimated_runtime_savings_percent": (
                    group["estimated_runtime_savings_percent"].mean()
                ),
                "mean_estimated_runtime_savings_seconds": (
                    group["estimated_runtime_savings_seconds"].mean()
                ),
            })

    return pd.DataFrame(rows)


def summarize_rankings(ranking):
    rows = []

    for model, model_df in ranking.groupby("meta_model", sort=True):
        for classifier in CLASSIFIERS + ["ALL"]:
            group = (
                model_df
                if classifier == "ALL"
                else model_df[model_df["classifier"] == classifier]
            )

            rows.append({
                "meta_model": model,
                "classifier": classifier,
                "dataset_classifier_cases": len(group),
                "top1_accuracy_percent": group["top1_correct"].mean() * 100.0,
                "top3_accuracy_percent": group["top3_correct"].mean() * 100.0,
                "mean_spearman": group["spearman"].mean(),
                "mean_kendall": group["kendall"].mean(),
            })

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = load_meta()

    all_details = []
    all_rankings = []
    global_choices = []

    print("=" * 75)
    print("ResampleRank - LOPGO Selection Policy Comparison")
    print("=" * 75)

    print("Meta-dataset rows:", len(meta))
    print("Datasets:", meta["dataset"].nunique())
    print("Each global baseline excludes its entire held-out parent group.")

    for model in MODELS:
        print("\nEvaluating policy comparisons for:", model)

        predictions = load_oof_predictions(meta, model)

        details, ranking, globals_df = evaluate_policies(
            meta, predictions, model
        )

        if len(details) != 688:
            raise RuntimeError(
                "{}: expected 688 policy rows; got {}.".format(
                    model, len(details)
                )
            )

        if len(ranking) != 172:
            raise RuntimeError(
                "{}: expected 172 ranking cases; got {}.".format(
                    model, len(ranking)
                )
            )

        all_details.append(details)
        all_rankings.append(ranking)
        global_choices.append(globals_df)

    detail_df = pd.concat(all_details, ignore_index=True)
    ranking_df = pd.concat(all_rankings, ignore_index=True)
    globals_df = pd.concat(global_choices, ignore_index=True)

    if len(detail_df) != 2752:
        raise RuntimeError(
            "Expected 2,752 policy evaluations; got {}.".format(
                len(detail_df)
            )
        )

    if len(ranking_df) != 688:
        raise RuntimeError(
            "Expected 688 ranking cases; got {}.".format(
                len(ranking_df)
            )
        )

    summary_df = summarize(detail_df, ranking_df)
    ranking_summary_df = summarize_rankings(ranking_df)

    output_files = {
        "lopgo_policy_details.csv": detail_df,
        "lopgo_policy_summary.csv": summary_df,
        "lopgo_raw_ranking_metrics.csv": ranking_df,
        "lopgo_raw_ranking_summary.csv": ranking_summary_df,
        "lopgo_global_strategy_choices.csv": globals_df.drop_duplicates(
            ["held_out_dataset", "classifier"]
        ),
    }

    for name, frame in output_files.items():
        frame.to_csv(OUTPUT_DIR / name, index=False)

    print("\n" + "=" * 75)
    print("LODO POLICY SUMMARY")
    print("=" * 75)

    print(
        summary_df[
            (summary_df["classifier"] == "ALL")
        ].to_string(index=False)
    )

    print("\nRAW RANKING SUMMARY")
    print(ranking_summary_df[
        ranking_summary_df["classifier"] == "ALL"
    ].to_string(index=False))

    print("\nSaved files:")
    for name in output_files:
        print(OUTPUT_DIR / name)

    print(
        "\nGlobal strategies were selected using only the other parent groups."
        "in each held-out fold."
    )
    print(
        "Runtime savings are estimates from recorded base-level runtimes "
        "and exclude meta-model inference overhead."
    )


if __name__ == "__main__":
    main()

