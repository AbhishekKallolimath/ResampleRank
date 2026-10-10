from pathlib import Path
import shutil

path = Path("src/final_reproducibility_audit.py")
content = path.read_text(encoding="utf-8-sig")

old_check = '''    record(
        str(relative_path) + " finite numeric values",
        bool(np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all()),
        "Numeric values checked for NaN/Infinity",
    )'''

new_check = '''    numeric_df = df.select_dtypes(include=[np.number])

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
    )'''

old_meta = 'meta = load_csv("results/meta_dataset_clean43.csv", 1204)'
new_meta = 'meta = load_csv("results/meta_dataset_clean43.csv", 1204, allow_failure_reason=True)'

if content.count(old_check) != 1:
    raise SystemExit("STOP: Expected audit check was not found exactly once.")

if content.count(old_meta) != 1:
    raise SystemExit("STOP: Expected meta-dataset call was not found exactly once.")

backup = path.with_name(path.name + ".before_false_positive_fix.bak")
if backup.exists():
    raise SystemExit("STOP: Audit-script backup already exists.")

shutil.copy2(str(path), str(backup))
content = content.replace(old_check, new_check, 1)
content = content.replace(old_meta, new_meta, 1)
path.write_text(content, encoding="utf-8")

print("Audit criteria corrected.")
print("Original audit script backed up to:", backup)
