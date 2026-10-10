import pandas as pd
from pathlib import Path

path = Path("results/selection_baselines_lodo_clean43/lodo_policy_details.csv")
df = pd.read_csv(path)

print("=" * 75)
print("POLICY DETAILS DATA INSPECTION")
print("=" * 75)

print("\nShape:", df.shape)
print("\nColumns:")
for column in df.columns:
    print(" -", column)

print("\nUnique values in key columns:")
for column in ["meta_model", "policy", "dataset", "classifier"]:
    if column in df.columns:
        print("\n" + column + ":")
        print(sorted(df[column].dropna().unique().tolist()))

print("\nFirst 8 rows:")
print(df.head(8).to_string(index=False))

print("\nData types:")
print(df.dtypes.to_string())
