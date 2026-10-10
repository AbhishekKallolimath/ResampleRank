import pandas as pd
import numpy as np
from pathlib import Path

folder = Path(
    "results/statistical_policy_comparison_lodo_clean43"
)

files = [
    "policy_details_adjusted_inference.csv",
    "policy_runtime_summary_adjusted_inference.csv",
    "policy_runtime_by_classifier_adjusted_inference.csv",
]

print("=" * 75)
print("ADJUSTED RUNTIME OUTPUT VALIDATION")
print("=" * 75)

for name in files:
    df = pd.read_csv(folder / name)
    print(
        "{} | rows={} | missing={} | duplicate rows={}".format(
            name,
            len(df),
            int(df.isna().sum().sum()),
            int(df.duplicated().sum()),
        )
    )

detail = pd.read_csv(folder / files[0])

assert len(detail) == 2752, "Unexpected detail row count."
assert not detail.isna().any().any(), "Missing values found."

keys = ["meta_model", "dataset", "classifier", "policy"]
assert not detail.duplicated(keys).any(), "Duplicate policy keys found."

expected_runtime = (
    detail["selected_runtime_seconds"]
    + detail["inference_overhead_applied_seconds"]
)
runtime_error = (
    detail["adjusted_selected_runtime_seconds"] - expected_runtime
).abs().max()

expected_savings = (
    detail["full_seven_strategy_runtime_seconds"]
    - detail["adjusted_selected_runtime_seconds"]
)
savings_error = (
    detail["adjusted_estimated_runtime_savings_seconds"]
    - expected_savings
).abs().max()

expected_percent = (
    100 * detail["adjusted_estimated_runtime_savings_seconds"]
    / detail["full_seven_strategy_runtime_seconds"]
)
percent_error = (
    detail["adjusted_estimated_runtime_savings_percent"]
    - expected_percent
).abs().max()

non_selector = detail[
    detail["policy"] != "meta_learning_baseline_aware"
]
assert (
    non_selector["inference_overhead_applied_seconds"] == 0
).all(), "Inference overhead incorrectly added to another policy."

assert runtime_error < 1e-9, "Adjusted runtime formula mismatch."
assert savings_error < 1e-9, "Savings formula mismatch."
assert percent_error < 1e-9, "Savings percentage mismatch."

summary = pd.read_csv(folder / files[1])
by_classifier = pd.read_csv(folder / files[2])

assert len(summary) == 16, "Expected 16 overall policy groups."
assert len(by_classifier) == 64, "Expected 64 classifier groups."

print("\nMaximum runtime formula error:", runtime_error)
print("Maximum savings formula error:", savings_error)
print("Maximum percentage formula error:", percent_error)
print(
    "Cases slower than full search:",
    int(detail["runtime_exceeds_full_search"].sum()),
)
print("\nPASS: adjusted runtime formulas and output structure verified.")

print("\nAdjusted savings by meta-model selector:")
cols = [
    "meta_model",
    "mean_original_savings_percent",
    "mean_adjusted_savings_percent",
]
selectors = summary[
    summary["policy"] == "meta_learning_baseline_aware"
][cols].copy()

selectors["reduction_percentage_points"] = (
    selectors["mean_original_savings_percent"]
    - selectors["mean_adjusted_savings_percent"]
)

print(selectors.to_string(index=False, float_format=lambda x: "{:.6f}".format(x)))
