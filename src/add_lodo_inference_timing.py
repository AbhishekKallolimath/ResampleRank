from pathlib import Path
import shutil

path = Path("src/train_meta_model_lodo_clean43.py")

raw = path.read_bytes()
had_bom = raw.startswith(b"\xef\xbb\xbf")
newline = "\r\n" if b"\r\n" in raw else "\n"
content = raw.decode("utf-8-sig").replace("\r\n", "\n")

replacements = [
    (
        "import numpy as np",
        "import time\nimport numpy as np",
    ),
    (
        "        predictions[\n"
        "            test_index\n"
        "        ] = pipeline.predict(\n"
        "            X_test\n"
        "        )",
        "        # Warm up the fitted pipeline before timing.\n"
        "        pipeline.predict(X_test)\n"
        "\n"
        "        inference_repeats = 30\n"
        "        inference_start = time.perf_counter()\n"
        "        for _ in range(inference_repeats):\n"
        "            fold_predictions = pipeline.predict(X_test)\n"
        "        total_inference_time_seconds = (\n"
        "            time.perf_counter() - inference_start\n"
        "        )\n"
        "\n"
        "        predictions[\n"
        "            test_index\n"
        "        ] = fold_predictions",
    ),
    (
        '                "test_rows": len(\n'
        "                    test_index\n"
        "                ),",
        '                "test_rows": len(\n'
        "                    test_index\n"
        "                ),\n"
        '                "inference_rows": len(X_test),\n'
        '                "inference_repeats": inference_repeats,\n'
        '                "total_inference_time_seconds": '
        "total_inference_time_seconds,\n"
        '                "mean_inference_time_seconds": (\n'
        "                    total_inference_time_seconds "
        "/ inference_repeats\n"
        "                ),",
    ),
]

for old, new in replacements:
    count = content.count(old)
    print("Expected one match; found:", count)
    if count != 1:
        raise SystemExit("STOPPED: source did not match; no changes written.")

backup = path.with_name(path.name + ".before_inference_timing.bak")
if backup.exists():
    raise SystemExit(
        "STOPPED: source backup already exists; no changes written."
    )

shutil.copy2(str(path), str(backup))

for old, new in replacements:
    content = content.replace(old, new, 1)

output = content.replace("\n", newline).encode("utf-8")
if had_bom:
    output = b"\xef\xbb\xbf" + output

path.write_bytes(output)

print("Updated:", path)
print("Source backup:", backup)
print("Inference timing added.")
