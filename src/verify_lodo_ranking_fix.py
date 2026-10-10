import pandas as pd
from pathlib import Path

meta = Path("results/meta_model_lodo_clean43")
policy = Path("results/selection_baselines_lodo_clean43")

print("=" * 75)
print("RANKING FILE CHECKS")
print("=" * 75)

ranking_files = sorted(meta.glob("*_ranking.csv"))
print("Ranking files found:", len(ranking_files))

for path in ranking_files:
    df = pd.read_csv(path)
    missing = int(df[["spearman", "kendall"]].isna().sum().sum())
    print(
        "{} | rows={} | missing correlations={} | duplicates={}".format(
            path.name, len(df), missing, int(df.duplicated().sum())
        )
    )

print("\n" + "=" * 75)
print("POLICY OUTPUT CHECKS")
print("=" * 75)

for name in [
    "lodo_policy_details.csv",
    "lodo_policy_summary.csv",
    "lodo_raw_ranking_metrics.csv",
    "lodo_raw_ranking_summary.csv",
    "lodo_global_strategy_choices.csv",
]:
    path = policy / name
    df = pd.read_csv(path)
    missing_corr = 0
    if "spearman" in df.columns:
        missing_corr += int(df["spearman"].isna().sum())
    if "kendall" in df.columns:
        missing_corr += int(df["kendall"].isna().sum())

    print(
        "{} | rows={} | missing cells={} | missing correlations={} | duplicates={}".format(
            name, len(df), int(df.isna().sum().sum()),
            missing_corr, int(df.duplicated().sum())
        )
    )

print("\n" + "=" * 75)
print("MODEL SUMMARY CHECKS")
print("=" * 75)

for name in ["meta_model_summary.csv", "meta_model_overall.csv"]:
    df = pd.read_csv(meta / name)
    print(
        "{} | rows={} | missing cells={} | duplicates={}".format(
            name, len(df), int(df.isna().sum().sum()),
            int(df.duplicated().sum())
        )
    )
