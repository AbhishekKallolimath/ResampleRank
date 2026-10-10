import pandas as pd
from pathlib import Path

base = Path("results")
timing_dir = base / "meta_model_lodo_clean43"
policy_file = (
    base / "selection_baselines_lodo_clean43"
    / "lodo_policy_details.csv"
)

records = []
problems = []

for path in sorted(timing_dir.glob("*_lodo_folds.csv")):
    folds = pd.read_csv(path)

    for _, row in folds.iterrows():
        names = str(row["test_datasets"]).split("|")

        if len(names) != 1:
            problems.append(
                "{} fold {} maps to {} datasets".format(
                    path.name, row["fold"], len(names)
                )
            )
            continue

        records.append({
            "meta_model": str(row["model"]),
            "dataset": names[0],
            "classifier": str(row["classifier"]),
            "inference_seconds": float(
                row["mean_inference_time_seconds"]
            ),
        })

timings = pd.DataFrame(records)
policies = pd.read_csv(policy_file)

policy_keys = policies[
    ["meta_model", "dataset", "classifier"]
].drop_duplicates()

print("=" * 75)
print("INFERENCE-TIMING MAPPING CHECK")
print("=" * 75)
print("Timing rows:", len(timings))
print("Unique policy cases:", len(policy_keys))
print("Duplicate timing keys:", int(timings.duplicated(
    ["meta_model", "dataset", "classifier"]
).sum()))

if problems:
    print("\nMAPPING PROBLEMS:")
    for problem in problems:
        print(problem)

merged = policy_keys.merge(
    timings,
    on=["meta_model", "dataset", "classifier"],
    how="outer",
    indicator=True,
    validate="one_to_one",
)

print("\nMatch status:")
print(merged["_merge"].value_counts().to_string())
print(
    "Missing inference times:",
    int(merged["inference_seconds"].isna().sum())
)

if len(timings):
    print("\nInference time by meta-model (milliseconds):")
    stats = timings.groupby("meta_model")["inference_seconds"].agg(
        ["count", "mean", "median", "max"]
    )
    stats[["mean", "median", "max"]] *= 1000
    print(stats.to_string(float_format=lambda x: "{:.4f}".format(x)))

if problems or (merged["_merge"] != "both").any():
    raise SystemExit("STOP: timing keys do not fully match policy cases.")

print("\nPASS: every policy case has a matching inference timing.")
