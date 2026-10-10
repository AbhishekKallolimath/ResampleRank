import pandas as pd
from pathlib import Path

folder = Path("results/meta_model_lodo_clean43")
files = sorted(folder.glob("*_lodo_folds.csv"))

print("=" * 75)
print("LODO INFERENCE TIMING VALIDATION")
print("=" * 75)
print("Fold files found:", len(files))

required = [
    "inference_rows",
    "inference_repeats",
    "total_inference_time_seconds",
    "mean_inference_time_seconds",
]

for path in files:
    df = pd.read_csv(path)
    print("\nFILE:", path.name)
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    missing_columns = [
        col for col in required if col not in df.columns
    ]

    if missing_columns:
        print("TIMING COLUMNS MISSING:", missing_columns)
        continue

    times = pd.to_numeric(
        df["mean_inference_time_seconds"], errors="coerce"
    )

    print("Missing timing values:", int(times.isna().sum()))
    print("Non-positive timing values:", int((times <= 0).sum()))
    print("Inference repeats:", sorted(
        df["inference_repeats"].dropna().unique().tolist()
    ))
    print("Mean per-batch inference (ms):", round(times.mean() * 1000, 6))
    print("Median per-batch inference (ms):", round(times.median() * 1000, 6))
    print("95th percentile per-batch inference (ms):",
          round(times.quantile(0.95) * 1000, 6))
    print("Estimated one-pass inference across recorded folds (s):",
          round(times.sum(), 6))

