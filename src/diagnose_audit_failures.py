import numpy as np
import pandas as pd
from pathlib import Path
from pandas.api.types import is_numeric_dtype

root = Path("results")
meta_file = root / "meta_dataset_clean43.csv"
prediction_dir = root / "meta_model_lodo_clean43"

def inspect_file(path):
    df = pd.read_csv(path)
    print("\n" + "=" * 80)
    print("FILE:", path)
    print("Shape:", df.shape)

    missing = df.isna().sum()
    missing = missing[missing > 0]
    print("\nColumns with missing values:")
    print(missing.to_string() if len(missing) else "None")

    bad_columns = []
    bad_masks = {}

    for col in df.columns:
        if not is_numeric_dtype(df[col]):
            continue

        values = df[col].to_numpy(dtype=float)
        nan_count = int(np.isnan(values).sum())
        inf_count = int(np.isinf(values).sum())

        if nan_count or inf_count:
            bad_columns.append((col, nan_count, inf_count))
            bad_masks[col] = ~np.isfinite(values)

    print("\nNumeric columns with non-finite values:")
    if not bad_columns:
        print("None")
    else:
        for col, nan_count, inf_count in bad_columns:
            print("{} | NaN={} | infinity={}".format(
                col, nan_count, inf_count
            ))

        bad_rows = np.zeros(len(df), dtype=bool)
        for mask in bad_masks.values():
            bad_rows |= mask

        key_cols = [
            c for c in [
                "dataset", "classifier", "resampling_code",
                "resampling", "target_macro_f1",
                "delta_macro_f1_vs_baseline"
            ] if c in df.columns
        ]
        bad_col_names = [c[0] for c in bad_columns]
        show_cols = list(dict.fromkeys(key_cols + bad_col_names))

        print("\nAffected rows (up to 10):")
        print(df.loc[bad_rows, show_cols].head(10).to_string(index=False))

inspect_file(meta_file)

prediction_files = sorted(prediction_dir.glob("*_predictions.csv"))
print("\nPrediction files found:", len(prediction_files))

for path in prediction_files:
    inspect_file(path)
