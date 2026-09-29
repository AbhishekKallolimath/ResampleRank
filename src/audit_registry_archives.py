
from pathlib import Path
import zipfile
import re
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
REGISTRY_FILE = BASE_DIR / "data" / "dataset_registry.csv"


def normalize(value):
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9]+", "", value)
    return value


def collect_files():
    files = []

    for path in RAW_DIR.rglob("*"):
        if not path.is_file():
            continue

        if "__MACOSX" in path.parts:
            continue

        if path.name.startswith("._"):
            continue

        files.append(path)

    return files


def inspect_zip(path):
    entries = []

    try:
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                clean = name.replace("\\", "/")

                if "__MACOSX/" in clean:
                    continue

                base = Path(clean).name

                if not base or base.startswith("._"):
                    continue

                entries.append(clean)

    except Exception:
        return []

    return entries


def main():
    registry = pd.read_csv(REGISTRY_FILE)

    files = collect_files()

    direct_files = []
    zip_files = []

    for path in files:
        if path.suffix.lower() == ".zip":
            zip_files.append(path)
        else:
            direct_files.append(path)

    print("=" * 80)
    print("RESAMPLERANK REGISTRY / ARCHIVE AUDIT")
    print("=" * 80)
    print("Registry datasets :", len(registry))
    print("Direct files      :", len(direct_files))
    print("ZIP files         :", len(zip_files))
    print()

    zip_contents = {}

    for zip_path in zip_files:
        zip_contents[zip_path] = inspect_zip(zip_path)

    for dataset in registry["dataset"].astype(str):
        dnorm = normalize(dataset)

        direct_matches = []
        zip_name_matches = []
        zip_content_matches = []

        for path in direct_files:
            stem_norm = normalize(path.stem)
            name_norm = normalize(path.name)

            if dnorm == stem_norm or dnorm == name_norm:
                direct_matches.append(path)

        for zip_path, entries in zip_contents.items():
            zip_stem_norm = normalize(zip_path.stem)

            if dnorm == zip_stem_norm:
                zip_name_matches.append(zip_path)

            for entry in entries:
                entry_norm = normalize(Path(entry).stem)
                entry_name_norm = normalize(Path(entry).name)

                if dnorm == entry_norm or dnorm == entry_name_norm:
                    zip_content_matches.append((zip_path, entry))

        if direct_matches:
            status = "READY_DIRECT"

        elif zip_content_matches:
            status = "READY_ZIP_CONTENT"

        elif zip_name_matches:
            status = "ZIP_FOUND"

        else:
            status = "NOT_EXACT_MATCH"

        print("=" * 80)
        print("DATASET :", dataset)
        print("STATUS  :", status)

        if direct_matches:
            print("DIRECT FILES:")
            for p in direct_matches:
                print("  ", p.relative_to(BASE_DIR))

        if zip_name_matches:
            print("ZIP NAME MATCHES:")
            for p in zip_name_matches:
                print("  ", p.relative_to(BASE_DIR))

        if zip_content_matches:
            print("ZIP CONTENT MATCHES:")
            for zip_path, entry in zip_content_matches:
                print(
                    "  ",
                    zip_path.relative_to(BASE_DIR),
                    "->",
                    entry
                )

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
