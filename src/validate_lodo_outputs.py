import pandas as pd
from pathlib import Path

folders = [
    Path("results/meta_model_lodo_clean43"),
    Path("results/selection_baselines_lodo_clean43"),
]

for folder in folders:
    print("\nFOLDER:", folder)
    for file in sorted(folder.glob("*.csv")):
        df = pd.read_csv(file)
        print(
            "{} | rows={} | columns={} | missing={} | duplicates={}".format(
                file.name,
                len(df),
                len(df.columns),
                int(df.isna().sum().sum()),
                int(df.duplicated().sum()),
            )
        )
