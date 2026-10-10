import pandas as pd
from pathlib import Path

folder = Path("results/meta_model_lodo_clean43")

checks = [
    ("ridge_logistic_regression_predictions.csv", ["glass6", "wisconsin"]),
    ("ridge_random_forest_predictions.csv", ["page-blocks0", "zoo-3"]),
]

for filename, datasets in checks:
    path = folder / filename
    df = pd.read_csv(path)

    print("\n" + "=" * 80)
    print("FILE:", filename)
    print("COLUMNS:", df.columns.tolist())

    columns = [
        col for col in df.columns
        if col.lower() in [
            "dataset", "classifier", "resampling",
            "resampling_code", "target_macro_f1",
            "baseline_macro_f1", "delta_macro_f1_vs_baseline",
            "predicted_delta_macro_f1", "actual_delta_macro_f1",
            "predicted_delta", "actual_delta",
        ]
        or "pred" in col.lower()
        or "macro_f1" in col.lower()
        or "delta" in col.lower()
        or "resampl" in col.lower()
    ]

    for dataset in datasets:
        subset = df[df["dataset"] == dataset]
        print("\nDATASET:", dataset, "| rows:", len(subset))
        if not subset.empty:
            print(subset[columns].to_string(index=False))
