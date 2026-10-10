from pathlib import Path
import pandas as pd

root = Path("figures/paper_clean43")
table_dir = root / "tables"

figure_names = [
    "fig01_meta_model_prediction_error",
    "fig02_strategy_selection_accuracy",
    "fig03_ranking_correlations",
    "fig04_policy_mean_improvement",
    "fig05_adjusted_runtime_savings",
    "fig06_classifier_robustness",
]

table_rows = {
    "table_meta_model_overall.csv": 4,
    "table_policy_summary_overall.csv": 16,
    "table_paired_wilcoxon_holm.csv": 8,
    "table_paired_mean_effects_holm.csv": 8,
    "table_runtime_savings_adjusted.csv": 4,
    "table_classifier_robustness_vs_global.csv": 16,
}

failures = []

print("=" * 75)
print("PAPER FIGURE VALIDATION")
print("=" * 75)

for name in figure_names:
    for ext in [".png", ".pdf"]:
        path = root / (name + ext)
        passed = path.exists() and path.stat().st_size > 0
        print("{} | {}".format(path.name, "PASS" if passed else "FAIL"))
        if not passed:
            failures.append(str(path))

print("\n" + "=" * 75)
print("PAPER TABLE VALIDATION")
print("=" * 75)

for name, expected_rows in table_rows.items():
    path = table_dir / name
    if not path.exists():
        print("{} | FAIL | file missing".format(name))
        failures.append(str(path))
        continue

    df = pd.read_csv(path)
    passed = len(df) == expected_rows and not df.isna().any().any()
    print("{} | rows={} | expected={} | {}".format(
        name, len(df), expected_rows, "PASS" if passed else "FAIL"
    ))
    if not passed:
        failures.append(str(path))

print("\nFinal status:", "PASS" if not failures else "FAIL")
if failures:
    print("Items needing attention:")
    for item in failures:
        print(" -", item)
