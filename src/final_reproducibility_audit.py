import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
META = RESULTS / "meta_model_lodo_clean43"
POLICY = RESULTS / "selection_baselines_lodo_clean43"
STATS = RESULTS / "statistical_policy_comparison_lodo_clean43"

checks = []


def record(name, passed, detail):
    checks.append({
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "detail": str(detail),
    })


def load_csv(relative_path, expected_rows=None, allow_failure_reason=False):
    path = ROOT / relative_path

    if not path.exists():
        record(str(relative_path) + " exists", False, "File not found")
        return None

    df = pd.read_csv(path)
    record(str(relative_path) + " exists", True, "File readable")

    if expected_rows is not None:
        record(
            str(relative_path) + " row count",
            len(df) == expected_rows,
            "Expected {}; found {}".format(expected_rows, len(df)),
        )

    if allow_failure_reason and "failure_reason" in df.columns:
        other = df.drop(columns=["failure_reason"])
        missing = int(other.isna().sum().sum())
        reason_blank = bool(df["failure_reason"].isna().all())

        record(
            str(relative_path) + " data completeness",
            missing == 0,
            "Missing cells outside failure_reason: {}".format(missing),
        )
        record(
            str(relative_path) + " failure_reason field",
            reason_blank,
            "All blank, as expected for successful predictions",
        )
    else:
        missing = int(df.isna().sum().sum())
        record(
            str(relative_path) + " missing values",
            missing == 0,
            "Missing cells: {}".format(missing),
        )

    numeric_df = df.select_dtypes(include=[np.number])

    # failure_reason is intentionally blank for successful experiments.
    # Exclude it from numeric finiteness checks after validating its status.
    if allow_failure_reason and "failure_reason" in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=["failure_reason"])

    finite_values = bool(
        np.isfinite(numeric_df.to_numpy()).all()
    )

    record(
        str(relative_path) + " finite numeric values",
        finite_values,
        "Numeric values checked for NaN/Infinity, excluding the blank failure_reason field",
    )

    return df


def check_unique(df, columns, name):
    if df is None or not set(columns).issubset(df.columns):
        record(name, False, "Required columns unavailable")
        return

    count = int(df.duplicated(columns).sum())
    record(name, count == 0, "Duplicate keys: {}".format(count))


print("=" * 78)
print("FINAL RESAMPLERANK REPRODUCIBILITY AUDIT")
print("=" * 78)

# 1. Meta-dataset
meta = load_csv("results/meta_dataset_clean43.csv", 1204, allow_failure_reason=True)

if meta is not None:
    meta_keys = ["dataset", "classifier", "resampling_code"]
    check_unique(meta, meta_keys, "Meta-dataset experiment keys")

    record(
        "Meta-dataset contains 43 datasets",
        meta["dataset"].nunique() == 43,
        "Found {}".format(meta["dataset"].nunique()),
    )

    record(
        "Meta-dataset contains four classifiers",
        meta["classifier"].nunique() == 4,
        "Found {}".format(meta["classifier"].nunique()),
    )

    record(
        "Meta-dataset contains seven strategies",
        meta["resampling_code"].nunique() == 7,
        "Found {}".format(meta["resampling_code"].nunique()),
    )

# 2. Predictions, rankings, and fold timing files
prediction_files = sorted(META.glob("*_predictions.csv"))
ranking_files = sorted(META.glob("*_ranking.csv"))
fold_files = sorted(META.glob("*_lodo_folds.csv"))

record("Prediction file count", len(prediction_files) == 16,
       "Found {}".format(len(prediction_files)))
record("Ranking file count", len(ranking_files) == 16,
       "Found {}".format(len(ranking_files)))
record("LODO fold file count", len(fold_files) == 4,
       "Found {}".format(len(fold_files)))

for path in prediction_files:
    df = load_csv(
        path.relative_to(ROOT),
        expected_rows=301,
        allow_failure_reason=True,
    )

    if df is not None:
        check_unique(
            df,
            ["dataset", "classifier", "resampling_code"],
            path.name + " experiment keys",
        )

        if "verification" in df.columns:
            passed = df["verification"].astype(str).str.upper().eq("PASS").all()
            record(path.name + " metric verification", passed,
                   "All rows must have PASS verification")

        if "metric_error" in df.columns:
            error = pd.to_numeric(df["metric_error"], errors="coerce")
            maximum = float(error.max()) if error.notna().all() else float("inf")
            record(path.name + " metric error tolerance",
                   maximum < 1e-6,
                   "Maximum absolute metric error: {}".format(maximum))

for path in ranking_files:
    df = load_csv(path.relative_to(ROOT), expected_rows=43)

    if df is not None and {"spearman", "kendall"}.issubset(df.columns):
        valid = (
            df["spearman"].notna().all()
            and df["kendall"].notna().all()
        )
        record(path.name + " ranking correlations", valid,
               "All Spearman/Kendall values must be present")

for path in fold_files:
    df = load_csv(path.relative_to(ROOT), expected_rows=172)

    if df is not None:
        required = [
            "inference_rows",
            "inference_repeats",
            "mean_inference_time_seconds",
            "test_datasets",
            "model",
            "classifier",
        ]

        has_columns = set(required).issubset(df.columns)
        record(path.name + " timing schema", has_columns,
               "Required inference timing columns")

        if has_columns:
            times = pd.to_numeric(
                df["mean_inference_time_seconds"], errors="coerce"
            )
            repeats = set(df["inference_repeats"].dropna().tolist())
            one_dataset = df["test_datasets"].astype(str).str.split("|").map(
                len
            ).eq(1).all()

            record(path.name + " positive inference times",
                   bool((times > 0).all()),
                   "All recorded times must be positive")
            record(path.name + " timing repeats",
                   repeats == {30},
                   "Observed repeat counts: {}".format(sorted(repeats)))
            record(path.name + " held-out dataset mapping", one_dataset,
                   "Each fold must identify exactly one dataset")

# 3. Meta-model summary files
load_csv("results/meta_model_lodo_clean43/meta_model_summary.csv", 20)
load_csv("results/meta_model_lodo_clean43/meta_model_overall.csv", 4)

# 4. Core policy-analysis outputs
policy_specs = [
    ("lodo_policy_details.csv", 2752),
    ("lodo_policy_summary.csv", 80),
    ("lodo_raw_ranking_metrics.csv", 688),
    ("lodo_raw_ranking_summary.csv", 20),
    ("lodo_global_strategy_choices.csv", 172),
]

policy_frames = {}

for name, rows in policy_specs:
    policy_frames[name] = load_csv(
        "results/selection_baselines_lodo_clean43/" + name, rows
    )

details = policy_frames["lodo_policy_details.csv"]

if details is not None:
    check_unique(
        details,
        ["meta_model", "dataset", "classifier", "policy"],
        "Policy-detail keys",
    )

raw_ranking = policy_frames["lodo_raw_ranking_metrics.csv"]

if raw_ranking is not None and {"spearman", "kendall"}.issubset(
    raw_ranking.columns
):
    valid = (
        raw_ranking["spearman"].notna().all()
        and raw_ranking["kendall"].notna().all()
    )
    record("Policy raw-ranking correlations", valid,
           "All 688 correlations must be present")

# 5. Statistical comparison files
stats_specs = [
    ("paired_policy_comparisons.csv", 8),
    ("paired_mean_effect_tests.csv", 8),
    ("policy_robustness_by_classifier.csv", 32),
    ("dataset_level_policy_values.csv", 688),
    ("policy_details_adjusted_inference.csv", 2752),
    ("policy_runtime_summary_adjusted_inference.csv", 16),
    ("policy_runtime_by_classifier_adjusted_inference.csv", 64),
]

stats_frames = {}

for name, rows in stats_specs:
    stats_frames[name] = load_csv(
        "results/statistical_policy_comparison_lodo_clean43/" + name,
        rows,
    )

for name in [
    "method_notes.txt",
    "runtime_adjustment_notes.txt",
]:
    path = STATS / name
    record(
        "Documentation file " + name,
        path.exists(),
        "Present" if path.exists() else "Missing",
    )

comparisons = stats_frames["paired_policy_comparisons.csv"]

if comparisons is not None:
    required = ["p_value", "holm_adjusted_p_value"]
    if set(required).issubset(comparisons.columns):
        valid = (
            comparisons[required].notna().all().all()
            and comparisons[required].ge(0).all().all()
            and comparisons[required].le(1).all().all()
        )
        record("Paired-test p-value ranges", valid,
               "All raw and adjusted p-values must lie in [0, 1]")

mean_tests = stats_frames["paired_mean_effect_tests.csv"]

if mean_tests is not None and {
    "p_value", "holm_adjusted_p_value"
}.issubset(mean_tests.columns):
    valid = (
        mean_tests[["p_value", "holm_adjusted_p_value"]].notna().all().all()
        and mean_tests[["p_value", "holm_adjusted_p_value"]].ge(0).all().all()
        and mean_tests[["p_value", "holm_adjusted_p_value"]].le(1).all().all()
    )
    record("Mean-effect test p-value ranges", valid,
           "All raw and adjusted p-values must lie in [0, 1]")

# 6. Match all recorded timing rows to policy cases
if details is not None:
    timing_records = []

    for path in fold_files:
        if not path.exists():
            continue

        folds = pd.read_csv(path)
        if not {
            "model", "classifier", "test_datasets",
            "mean_inference_time_seconds",
        }.issubset(folds.columns):
            continue

        for _, row in folds.iterrows():
            names = str(row["test_datasets"]).split("|")
            if len(names) == 1:
                timing_records.append({
                    "meta_model": str(row["model"]),
                    "dataset": names[0],
                    "classifier": str(row["classifier"]),
                    "mean_inference_time_seconds": float(
                        row["mean_inference_time_seconds"]
                    ),
                })

    timings = pd.DataFrame(timing_records)
    keys = ["meta_model", "dataset", "classifier"]
    policy_keys = details[keys].drop_duplicates()

    timing_ok = (
        len(timings) == 688
        and not timings.duplicated(keys).any()
    )

    if timing_ok:
        mapped = policy_keys.merge(
            timings,
            on=keys,
            how="outer",
            indicator=True,
            validate="one_to_one",
        )
        timing_ok = bool(mapped["_merge"].eq("both").all())

    record("Inference timing matches policy cases",
           timing_ok,
           "Expected 688 unique one-to-one dataset/classifier/model matches")

# 7. Adjusted runtime arithmetic
adjusted = stats_frames["policy_details_adjusted_inference.csv"]

if adjusted is not None:
    required = [
        "selected_runtime_seconds",
        "inference_overhead_applied_seconds",
        "adjusted_selected_runtime_seconds",
        "full_seven_strategy_runtime_seconds",
        "adjusted_estimated_runtime_savings_seconds",
        "adjusted_estimated_runtime_savings_percent",
        "policy",
    ]

    if set(required).issubset(adjusted.columns):
        runtime_error = (
            adjusted["adjusted_selected_runtime_seconds"]
            - adjusted["selected_runtime_seconds"]
            - adjusted["inference_overhead_applied_seconds"]
        ).abs().max()

        savings_error = (
            adjusted["adjusted_estimated_runtime_savings_seconds"]
            - (
                adjusted["full_seven_strategy_runtime_seconds"]
                - adjusted["adjusted_selected_runtime_seconds"]
            )
        ).abs().max()

        percent = (
            100.0
            * adjusted["adjusted_estimated_runtime_savings_seconds"]
            / adjusted["full_seven_strategy_runtime_seconds"]
        )

        percent_error = (
            adjusted["adjusted_estimated_runtime_savings_percent"]
            - percent
        ).abs().max()

        record("Adjusted selected-runtime formula",
               float(runtime_error) < 1e-9,
               "Maximum error: {}".format(runtime_error))
        record("Adjusted savings formula",
               float(savings_error) < 1e-9,
               "Maximum error: {}".format(savings_error))
        record("Adjusted savings percentage formula",
               float(percent_error) < 1e-9,
               "Maximum error: {}".format(percent_error))

        other_policies = adjusted[
            adjusted["policy"] != "meta_learning_baseline_aware"
        ]
        zero_overhead = (
            other_policies["inference_overhead_applied_seconds"] == 0
        ).all()

        record("Inference overhead applied only to learned selector",
               bool(zero_overhead),
               "Other policy overhead must remain zero")

# 8. Save the audit manifest
audit = pd.DataFrame(checks)
audit_path = RESULTS / "final_reproducibility_audit.csv"
audit.to_csv(audit_path, index=False)

failures = audit[audit["status"] == "FAIL"]

print("\n" + "=" * 78)
print("AUDIT RESULTS")
print("=" * 78)
print("Checks:", len(audit))
print("Passed:", int((audit["status"] == "PASS").sum()))
print("Failed:", len(failures))
print("Audit report:", audit_path)

if not failures.empty:
    print("\nFAILED CHECKS:")
    print(failures.to_string(index=False))
else:
    print("\nPASS: all final reproducibility checks succeeded.")

