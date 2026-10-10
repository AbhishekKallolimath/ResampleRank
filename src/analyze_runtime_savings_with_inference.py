import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"

POLICY_FILE = (
    RESULTS / "selection_baselines_lodo_clean43"
    / "lodo_policy_details.csv"
)
TIMING_DIR = RESULTS / "meta_model_lodo_clean43"
OUTPUT_DIR = (
    RESULTS / "statistical_policy_comparison_lodo_clean43"
)

KEYS = ["meta_model", "dataset", "classifier"]
META_POLICY = "meta_learning_baseline_aware"


def main():
    policies = pd.read_csv(POLICY_FILE)

    if len(policies) != 2752:
        raise ValueError(
            "Expected 2752 policy rows, found {}".format(len(policies))
        )

    timing_records = []

    for path in sorted(TIMING_DIR.glob("*_lodo_folds.csv")):
        folds = pd.read_csv(path)

        required = [
            "model", "classifier", "test_datasets",
            "mean_inference_time_seconds",
        ]
        if any(col not in folds.columns for col in required):
            raise ValueError("Required timing columns missing: " + path.name)

        for _, row in folds.iterrows():
            datasets = str(row["test_datasets"]).split("|")

            if len(datasets) != 1:
                raise ValueError(
                    "Expected one held-out dataset in {} fold {}".format(
                        path.name, row["fold"]
                    )
                )

            timing_records.append({
                "meta_model": str(row["model"]),
                "dataset": datasets[0],
                "classifier": str(row["classifier"]),
                "meta_model_inference_seconds": float(
                    row["mean_inference_time_seconds"]
                ),
            })

    timings = pd.DataFrame(timing_records)

    if len(timings) != 688:
        raise ValueError(
            "Expected 688 timing rows, found {}".format(len(timings))
        )

    if timings.duplicated(KEYS).any():
        raise ValueError("Duplicate timing keys detected.")

    merged = policies.merge(
        timings,
        on=KEYS,
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if not (merged["_merge"] == "both").all():
        raise ValueError("Some policy rows have no inference timing.")

    merged.drop(columns=["_merge"], inplace=True)

    numeric_cols = [
        "meta_model_inference_seconds",
        "selected_runtime_seconds",
        "full_seven_strategy_runtime_seconds",
    ]

    if merged[numeric_cols].isna().any().any():
        raise ValueError("Missing runtime values found.")

    if (merged[numeric_cols] < 0).any().any():
        raise ValueError("Negative runtime values detected.")

    if (merged["full_seven_strategy_runtime_seconds"] <= 0).any():
        raise ValueError("Full-search runtime must be positive.")

    # Only the learned selector incurs model inference overhead.
    merged["inference_overhead_applied_seconds"] = np.where(
        merged["policy"] == META_POLICY,
        merged["meta_model_inference_seconds"],
        0.0,
    )

    merged["adjusted_selected_runtime_seconds"] = (
        merged["selected_runtime_seconds"]
        + merged["inference_overhead_applied_seconds"]
    )

    merged["adjusted_estimated_runtime_savings_seconds"] = (
        merged["full_seven_strategy_runtime_seconds"]
        - merged["adjusted_selected_runtime_seconds"]
    )

    merged["adjusted_estimated_runtime_savings_percent"] = (
        100.0
        * merged["adjusted_estimated_runtime_savings_seconds"]
        / merged["full_seven_strategy_runtime_seconds"]
    )

    merged["runtime_exceeds_full_search"] = (
        merged["adjusted_estimated_runtime_savings_seconds"] < 0
    )

    detail_file = OUTPUT_DIR / "policy_details_adjusted_inference.csv"
    merged.to_csv(detail_file, index=False)

    group_keys = ["meta_model", "policy"]

    def summarize(frame, keys):
        summary = frame.groupby(keys, as_index=False).agg(
            cases=("dataset", "size"),
            datasets=("dataset", "nunique"),
            mean_improvement_vs_baseline=(
                "selected_improvement_vs_baseline", "mean"
            ),
            mean_regret=("regret", "mean"),
            resampling_selection_rate_percent=(
                "used_resampling", lambda x: 100.0 * x.mean()
            ),
            mean_inference_overhead_applied_seconds=(
                "inference_overhead_applied_seconds", "mean"
            ),
            mean_selected_runtime_seconds=(
                "selected_runtime_seconds", "mean"
            ),
            mean_adjusted_selected_runtime_seconds=(
                "adjusted_selected_runtime_seconds", "mean"
            ),
            mean_full_seven_strategy_runtime_seconds=(
                "full_seven_strategy_runtime_seconds", "mean"
            ),
            mean_original_savings_percent=(
                "estimated_runtime_savings_percent", "mean"
            ),
            mean_adjusted_savings_seconds=(
                "adjusted_estimated_runtime_savings_seconds", "mean"
            ),
            mean_adjusted_savings_percent=(
                "adjusted_estimated_runtime_savings_percent", "mean"
            ),
            cases_slower_than_full_search=(
                "runtime_exceeds_full_search", "sum"
            ),
        )
        return summary

    overall = summarize(merged, group_keys)
    overall.to_csv(
        OUTPUT_DIR / "policy_runtime_summary_adjusted_inference.csv",
        index=False,
    )

    by_classifier = summarize(
        merged, ["meta_model", "policy", "classifier"]
    )
    by_classifier.to_csv(
        OUTPUT_DIR / "policy_runtime_by_classifier_adjusted_inference.csv",
        index=False,
    )

    notes = (
        "Adjusted runtime = selected strategy runtime + measured "
        "meta-model inference overhead for meta_learning_baseline_aware.\n"
        "Other policies receive zero meta-model inference overhead; "
        "oracle_upper_bound is theoretical.\n"
        "Inference timing includes pipeline prediction and preprocessing "
        "for seven candidate rows, based on the recorded warm-up and "
        "30-repeat timing procedure.\n"
        "These estimates exclude dataset meta-feature extraction, model "
        "training, and additional policy-selection logic.\n"
        "Timings are hardware- and environment-dependent.\n"
        "Negative adjusted savings mean the estimated selected runtime "
        "exceeds the full seven-strategy runtime.\n"
    )
    (OUTPUT_DIR / "runtime_adjustment_notes.txt").write_text(
        notes, encoding="utf-8"
    )

    print("=" * 78)
    print("RUNTIME SAVINGS WITH INFERENCE OVERHEAD")
    print("=" * 78)
    print(
        overall.to_string(
            index=False,
            float_format=lambda x: "{:.6f}".format(x),
        )
    )

    print("\nDetail rows:", len(merged))
    print("Timing matches:", len(merged))
    print(
        "Cases slower than full search:",
        int(merged["runtime_exceeds_full_search"].sum()),
    )
    print("\nSaved files:")
    print(detail_file)
    print(OUTPUT_DIR / "policy_runtime_summary_adjusted_inference.csv")
    print(OUTPUT_DIR / "policy_runtime_by_classifier_adjusted_inference.csv")
    print(OUTPUT_DIR / "runtime_adjustment_notes.txt")


if __name__ == "__main__":
    main()
