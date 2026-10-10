import pandas as pd
from pathlib import Path

folders = [
    Path("results/meta_model_lodo_clean43"),
    Path("results/selection_baselines_lodo_clean43"),
]

for folder in folders:
    print("\n" + "=" * 80)
    print("FOLDER:", folder)
    print("=" * 80)

    for file in sorted(folder.glob("*.csv")):
        df = pd.read_csv(file)
        missing = df.isna().sum()
        missing = missing[missing > 0]

        if len(missing) == 0:
            continue

        print("\nFILE:", file.name)
        print("Missing values by column:")
        print(missing.to_string())

        if "predictions" not in file.name:
            affected = df[df.isna().any(axis=1)]
            print("Affected rows (up to 8):")
            print(affected.head(8).to_string(index=False))
