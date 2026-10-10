from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"

META_FILE = RESULTS_DIR / "meta_dataset_clean43.csv"
PREDICTION_FILE = (
    RESULTS_DIR
    / "meta_model_clean43"
    / "test_predictions.csv"
)
OUTPUT_DIR = RESULTS_DIR / "baseline_aware_selection_clean43"

CLASSIFIERS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
]

STRATEGY_CODES = [
    "none",
    "ros",
    "rus",
    "smote",
    "adasyn",
    "smote_tomek",
    "smote_enn",
]

POLICIES = [
    "always_no_resampling",
    "training_only_global_best",
    "meta_learning_baseline_aware",
    "oracle_upper_bound",
]


def load_inputs():
    for path in [META_FILE, PREDICTION_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                "Required input file not found: {}".format(path)
            )

    meta = pd.read_csv(META_FILE)
    predictions = pd.read_csv(PREDICTION_FILE)

    meta_required = [
        "dataset",
        "split",
        "classifier",
        "resampling_code",
        "resampling",
        "target_macro_f1",
        "baseline_macro_f1",
        "execution_time_seconds",
    ]

    pred_required = [
        "dataset",
        "split",
        "classifier",
        "resampling_code",
        "predicted_delta_macro_f1",
        "predicted_macro_f1",
    ]

    for label, frame, required in [
        ("Meta-dataset", meta, meta_required),
        ("Prediction file", predictions, pred_required),
    ]:
        missing = [
            column for column in required
            if column not in frame.columns
        ]
        if missing:
            raise ValueError(
                "{} is missing columns: {}".format(label, missing)
            )

    meta["split"] = meta["split"].astype(str).str.strip().str.upper()
    predictions["split"] = (
        predictions["split"].astype(str).str.strip().str.upper()
    )

    if len(meta) != 1204 or meta["dataset"].nunique() != 43:
        raise ValueError(
            "Expected 1,204 meta-dataset rows across 43 datasets."
        )

    if len(predictions) != 168:
        raise ValueError(
            "Expected 168 held-out predictions; found {}.".format(
                len(predictions)
            )
        )

    if predictions["dataset"].nunique() != 6:
        raise ValueError(
            "Prediction file must contain exactly six test datasets."
        )

    if not predictions["split"].eq("TEST").all():
        raise ValueError(
            "Prediction file contains non-test datasets."
        )

    keys = ["dataset", "classifier", "resampling_code"]

    if meta.duplicated(keys).any():
        raise ValueError(
            "Duplicate experiment keys in meta-dataset."
        )

    if predictions.duplicated(keys).any():
        raise ValueError(
            "Duplicate experiment keys in predictions."
        )

    test_meta = meta[meta["split"] == "TEST"].copy()

    if len(test_meta) != 168:
        raise ValueError(
            "Expected 168 actual test rows; found {}.".format(
                len(test_meta)
            )
        )

    prediction_values = predictions[
        keys + ["predicted_delta_macro_f1", "predicted_macro_f1"]
    ].copy()

    test = test_meta.merge(
        prediction_values,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    if test["predicted_delta_macro_f1"].isna().any():
        raise ValueError(
            "Some actual test rows lack predicted deltas."
        )

    if test["predicted_macro_f1"].isna().any():
        raise ValueError(
            "Some actual test rows lack predicted Macro-F1."
        )

    if not np.isfinite(
        test[
            [
                "target_macro_f1",
                "baseline_macro_f1",
                "execution_time_seconds",
                "predicted_delta_macro_f1",
                "predicted_macro_f1",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Actual values, runtimes, or predictions contain non-finite values."
        )

    return meta, test


def select_training_only_global_best(meta):
    train = meta[meta["split"] == "TRAIN"].copy()

    if train["dataset"].nunique() != 23:
        raise ValueError(
            "Global strategy selection must use only 23 training datasets."
        )

    means = (
        train.groupby(
            ["classifier", "resampling_code"],
            as_index=False,
        )
        ["target_macro_f1"]
        .mean()
        .rename(
            columns={
                "target_macro_f1": "training_mean_macro_f1"
            }
        )
    )

    best_rows = (
        means.sort_values(
            [
                "classifier",
                "training_mean_macro_f1",
                "resampling_code",
            ],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .groupby("classifier", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )

    best_map = dict(
        zip(
            best_rows["classifier"],
            best_rows["resampling_code"],
        )
    )

    return best_rows, best_map


def calculate_rank_metrics(group):
    actual = group["target_macro_f1"].to_numpy(dtype=float)
    predicted = group["predicted_macro_f1"].to_numpy(dtype=float)

    best_score = float(actual.max())
    best_codes = set(
        group.loc[
            np.isclose(actual, best_score, rtol=0.0, atol=1e-12),
            "resampling_code",
        ]
    )

    ranked = group.sort_values(
        ["predicted_macro_f1", "resampling_code"],
        ascending=[False, True],
        kind="mergesort",
    )

    predicted_best = ranked.iloc[0]["resampling_code"]
    predicted_top3 = set(
        ranked.head(3)["resampling_code"]
    )

    if len(np.unique(actual)) > 1 and len(np.unique(predicted)) > 1:
        sp = spearmanr(actual, predicted)[0]
        kt = kendalltau(actual, predicted)[0]
    else:
        sp = np.nan
        kt = np.nan

    return {
        "dataset": group.iloc[0]["dataset"],
        "classifier": group.iloc[0]["classifier"],
        "actual_best_codes": "|".join(sorted(best_codes)),
        "raw_predicted_best_code": predicted_best,
        "raw_prediction_top1_correct": predicted_best in best_codes,
        "raw_prediction_top3_correct": bool(
            predicted_top3.intersection(best_codes)
        ),
        "spearman": float(sp) if np.isfinite(sp) else np.nan,
        "kendall": float(kt) if np.isfinite(kt) else np.nan,
    }


def evaluate_policies(meta, test, global_best_map):
    details = []
    rank_rows = []

    for (dataset, classifier), group in test.groupby(
        ["dataset", "classifier"],
        sort=True,
    ):
        group = group.copy()

        if len(group) != 7 or group["resampling_code"].nunique() != 7:
            raise ValueError(
                "{} / {} must have seven strategies.".format(
                    dataset, classifier
                )
            )

        baseline_rows = group[group["resampling_code"] == "none"]

        if len(baseline_rows) != 1:
            raise ValueError(
                "{} / {} must have exactly one baseline row.".format(
                    dataset, classifier
                )
            )

        baseline = float(
            baseline_rows.iloc[0]["baseline_macro_f1"]
        )

        if not np.isclose(
            float(baseline_rows.iloc[0]["target_macro_f1"]),
            baseline,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError(
                "No-resampling score differs from its baseline for {} / {}.".format(
                    dataset, classifier
                )
            )

        actual_best_score = float(group["target_macro_f1"].max())
        actual_best_codes = set(
            group.loc[
                np.isclose(
                    group["target_macro_f1"],
                    actual_best_score,
                    rtol=0.0,
                    atol=1e-12,
                ),
                "resampling_code",
            ]
        )

        full_runtime = float(group["execution_time_seconds"].sum())

        if full_runtime <= 0:
            raise ValueError(
                "Non-positive total runtime for {} / {}.".format(
                    dataset, classifier
                )
            )

        non_none_positive = group[
            (group["resampling_code"] != "none")
            & (group["predicted_delta_macro_f1"] > 0)
        ]

        if non_none_positive.empty:
            learned_code = "none"
        else:
            learned_code = (
                non_none_positive.sort_values(
                    ["predicted_delta_macro_f1", "resampling_code"],
                    ascending=[False, True],
                    kind="mergesort",
                )
                .iloc[0]["resampling_code"]
            )

        raw_rank = calculate_rank_metrics(group)
        rank_rows.append(raw_rank)

        oracle_code = sorted(actual_best_codes)[0]

        selected_codes = {
            "always_no_resampling": "none",
            "training_only_global_best": global_best_map[classifier],
            "meta_learning_baseline_aware": learned_code,
            "oracle_upper_bound": oracle_code,
        }

        for policy, selected_code in selected_codes.items():
            selected_rows = group[
                group["resampling_code"] == selected_code
            ]

            if len(selected_rows) != 1:
                raise ValueError(
                    "Policy {} selected unknown or duplicate strategy {}.".format(
                        policy, selected_code
                    )
                )

            selected_row = selected_rows.iloc[0]
            selected_score = float(selected_row["target_macro_f1"])
            selected_runtime = float(
                selected_row["execution_time_seconds"]
            )

            details.append({
                "dataset": dataset,
                "classifier": classifier,
                "policy": policy,
                "selected_resampling_code": selected_code,
                "selected_resampling": selected_row["resampling"],
                "baseline_macro_f1": baseline,
                "selected_macro_f1": selected_score,
                "oracle_best_macro_f1": actual_best_score,
                "selected_improvement_vs_baseline": (
                    selected_score - baseline
                ),
                "oracle_improvement_vs_baseline": (
                    actual_best_score - baseline
                ),
                "regret": actual_best_score - selected_score,
                "matches_oracle": selected_code in actual_best_codes,
                "selected_runtime_seconds": selected_runtime,
                "full_seven_strategy_runtime_seconds": full_runtime,
                "estimated_runtime_savings_seconds": (
                    full_runtime - selected_runtime
                ),
                "estimated_runtime_savings_percent": (
                    (full_runtime - selected_runtime)
                    / full_runtime
                    * 100.0
                ),
                "used_resampling": selected_code != "none",
            })

    return pd.DataFrame(details), pd.DataFrame(rank_rows)


def summarize_results(details, ranking):
    rows = []

    for policy in POLICIES:
        policy_df = details[details["policy"] == policy]

        for classifier in CLASSIFIERS + ["ALL"]:
            if classifier == "ALL":
                group = policy_df
                rank_group = ranking
            else:
                group = policy_df[
                    policy_df["classifier"] == classifier
                ]
                rank_group = ranking[
                    ranking["classifier"] == classifier
                ]

            if group.empty:
                continue

            rows.append({
                "policy": policy,
                "classifier": classifier,
                "test_datasets": group["dataset"].nunique(),
                "dataset_classifier_cases": len(group),
                "mean_selected_macro_f1": group[
                    "selected_macro_f1"
                ].mean(),
                "mean_improvement_vs_baseline": group[
                    "selected_improvement_vs_baseline"
                ].mean(),
                "mean_regret": group["regret"].mean(),
                "oracle_match_rate_percent": (
                    group["matches_oracle"].mean() * 100.0
                ),
                "resampling_selection_rate_percent": (
                    group["used_resampling"].mean() * 100.0
                ),
                "mean_estimated_runtime_savings_percent": group[
                    "estimated_runtime_savings_percent"
                ].mean(),
                "mean_estimated_runtime_savings_seconds": group[
                    "estimated_runtime_savings_seconds"
                ].mean(),
                "raw_prediction_top1_accuracy": (
                    rank_group["raw_prediction_top1_correct"].mean()
                    * 100.0
                ),
                "raw_prediction_top3_accuracy": (
                    rank_group["raw_prediction_top3_correct"].mean()
                    * 100.0
                ),
                "raw_prediction_mean_spearman": rank_group[
                    "spearman"
                ].mean(),
                "raw_prediction_mean_kendall": rank_group[
                    "kendall"
                ].mean(),
            })

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("ResampleRank - Clean-43 Baseline Policy Comparison")
    print("=" * 75)

    meta, test = load_inputs()
    global_rows, global_best_map = select_training_only_global_best(meta)

    print("\nGlobal best strategies selected from TRAIN data only:")
    print(
        global_rows.sort_values("classifier").to_string(index=False)
    )

    details, ranking = evaluate_policies(
        meta,
        test,
        global_best_map,
    )

    if len(details) != 96:
        raise RuntimeError(
            "Expected 96 policy evaluations; found {}.".format(
                len(details)
            )
        )

    if details.duplicated(
        ["dataset", "classifier", "policy"]
    ).any():
        raise RuntimeError(
            "Duplicate dataset/classifier/policy evaluations found."
        )

    if len(ranking) != 24:
        raise RuntimeError(
            "Expected 24 raw ranking evaluations; found {}.".format(
                len(ranking)
            )
        )

    global_file = (
        OUTPUT_DIR / "training_only_global_strategies.csv"
    )
    details_file = (
        OUTPUT_DIR / "baseline_policy_details.csv"
    )
    ranking_file = (
        OUTPUT_DIR / "raw_prediction_ranking_metrics.csv"
    )
    summary_file = (
        OUTPUT_DIR / "baseline_policy_summary.csv"
    )

    summary = summarize_results(details, ranking)

    global_rows.to_csv(global_file, index=False)
    details.to_csv(details_file, index=False)
    ranking.to_csv(ranking_file, index=False)
    summary.to_csv(summary_file, index=False)

    print("\n" + "=" * 75)
    print("BASELINE POLICY SUMMARY")
    print("=" * 75)
    print(summary.to_string(index=False))

    print("\nSaved files:")
    print(global_file)
    print(details_file)
    print(ranking_file)
    print(summary_file)

    print("\nCompleted: all policy comparisons use the same six test datasets.")
    print("The global-best strategy was determined using TRAIN data only.")
    print("Runtime savings exclude meta-model training and inference overhead.")


if __name__ == "__main__":
    main()
