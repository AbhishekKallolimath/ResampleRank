from pathlib import Path
import shutil

plans = [
    (
        Path("src/train_meta_model_lodo_clean43.py"),
        [
            (
                '    Return rows ordered by predicted Macro-F1 descending.',
                '    Return rows ordered by predicted improvement descending.',
            ),
            (
                '        by=[\n'
                '            "predicted_macro_f1",\n'
                '            "resampling_code",\n'
                '        ],',
                '        by=[\n'
                '            "predicted_delta_macro_f1",\n'
                '            "resampling_code",\n'
                '        ],',
            ),
            (
                '    predicted_values = group[\n'
                '        "predicted_macro_f1"\n'
                '    ].to_numpy()',
                '    predicted_values = group[\n'
                '        "predicted_delta_macro_f1"\n'
                '    ].to_numpy()',
            ),
        ],
    ),
    (
        Path("src/analyze_selection_baselines_lodo_clean43.py"),
        [
            (
                '    predicted = group["predicted_macro_f1"].to_numpy(dtype=float)',
                '    predicted = group["predicted_delta_macro_f1"].to_numpy(dtype=float)',
            ),
            (
                '        ["predicted_macro_f1", "resampling_code"],',
                '        ["predicted_delta_macro_f1", "resampling_code"],',
            ),
        ],
    ),
]

loaded = []

for path, replacements in plans:
    if not path.exists():
        raise SystemExit("FILE NOT FOUND: " + str(path))

    raw = path.read_bytes()
    had_bom = raw.startswith(b"\xef\xbb\xbf")
    newline = "\r\n" if b"\r\n" in raw else "\n"
    content = raw.decode("utf-8-sig").replace("\r\n", "\n")

    for old, new in replacements:
        count = content.count(old)
        print("{} | expected 1 match | found {}".format(path.name, count))
        if count != 1:
            raise SystemExit(
                "STOPPED: Unexpected source content. No files changed."
            )

    loaded.append((path, content, replacements, had_bom, newline, raw))

for path, content, replacements, had_bom, newline, raw in loaded:
    backup = path.with_name(path.name + ".before_delta_ranking_fix.bak")

    if not backup.exists():
        shutil.copy2(str(path), str(backup))

    for old, new in replacements:
        content = content.replace(old, new, 1)

    output = content.replace("\n", newline).encode("utf-8")
    if had_bom:
        output = b"\xef\xbb\xbf" + output

    path.write_bytes(output)
    print("UPDATED:", path)
    print("BACKUP :", backup)

print("\nSource corrections completed.")
