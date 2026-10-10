from pathlib import Path

path = Path("src/validate_paper_outputs.py")
text = path.read_text(encoding="utf-8-sig")

old = '"table_meta_model_overall.csv": 20,'
new = '"table_meta_model_overall.csv": 4,'

if text.count(old) != 1:
    raise SystemExit("Expected entry not found exactly once; no changes made.")

path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("Validator corrected: the overall meta-model table should have 4 rows.")
