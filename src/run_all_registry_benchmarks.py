
from pathlib import Path
import argparse
import concurrent.futures
import subprocess
import sys
import time

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_FILE = PROJECT_ROOT / "data" / "dataset_registry.csv"
BENCHMARK_SCRIPT = PROJECT_ROOT / "src" / "run_registry_benchmark.py"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_registry():
    df = pd.read_csv(REGISTRY_FILE)

    required_columns = {"dataset", "split", "source_file"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Missing required registry columns: "
            + ", ".join(sorted(missing))
        )

    return df


def result_file(dataset_name):
    return RESULTS_DIR / f"{dataset_name}_all_results.csv"


def is_complete(dataset_name):
    path = result_file(dataset_name)

    if not path.exists():
        return False

    try:
        df = pd.read_csv(path)

        if len(df) != 28:
            return False

        if "Verification" in df.columns:
            verification = df["Verification"].astype(str).str.upper()
            if not verification.isin(["PASS", "NOT_APPLICABLE"]).all():
                return False

        return True

    except Exception:
        return False


def benchmark_dataset(dataset_name):
    start = time.time()

    command = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--dataset",
        dataset_name,
    ]

    try:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        elapsed = time.time() - start

        return {
            "dataset": dataset_name,
            "return_code": completed.returncode,
            "elapsed_seconds": round(elapsed, 3),
            "output": completed.stdout,
            "success": completed.returncode == 0 and is_complete(dataset_name),
        }

    except Exception as exc:
        return {
            "dataset": dataset_name,
            "return_code": -1,
            "elapsed_seconds": round(time.time() - start, 3),
            "output": str(exc),
            "success": False,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Run ResampleRank registry benchmarks automatically."
    )

    parser.add_argument(
        "--split",
        choices=["TRAIN", "VALIDATION", "TEST", "ALL"],
        default="ALL",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    if not REGISTRY_FILE.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_FILE}")

    if not BENCHMARK_SCRIPT.exists():
        raise FileNotFoundError(
            f"Benchmark script not found: {BENCHMARK_SCRIPT}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    registry = load_registry()

    total_registry_datasets = len(registry)

    mapped_registry = registry[
        registry["source_file"].notna()
        & (registry["source_file"].astype(str).str.strip() != "")
    ].copy()

    unmapped_registry = registry[
        registry["source_file"].isna()
        | (registry["source_file"].astype(str).str.strip() == "")
    ].copy()

    registry = mapped_registry

    if args.split != "ALL":
        registry = registry[
            registry["split"].astype(str).str.upper() == args.split
        ]

    datasets = registry["dataset"].astype(str).tolist()

    pending = []
    skipped = []

    for dataset_name in datasets:
        if is_complete(dataset_name):
            skipped.append(dataset_name)
        else:
            pending.append(dataset_name)

    if args.limit is not None:
        pending = pending[:args.limit]

    print("=" * 75)
    print("ResampleRank - Batch Registry Benchmark")
    print("=" * 75)
    print(f"Registry            : {REGISTRY_FILE}")
    print(f"Requested split     : {args.split}")
    print(f"Workers             : {args.workers}")
    print(f"Total registry      : {total_registry_datasets}")
    print(f"Mapped datasets     : {len(mapped_registry)}")
    print(f"Unmapped datasets   : {len(unmapped_registry)}")
    print(f"Already complete    : {len(skipped)}")
    print(f"Pending             : {len(pending)}")
    print(f"Benchmark script    : {BENCHMARK_SCRIPT}")
    print("=" * 75)

    if not unmapped_registry.empty:
        print("\nSkipped datasets without source mappings:")
    for dataset_name in unmapped_registry["dataset"].astype(str):
        print(f"  [NO SOURCE] {dataset_name}")

    if skipped:
        print("\nSkipped completed datasets:")
        for dataset_name in skipped:
            print(f"  [SKIP] {dataset_name}")

    if not pending:
        print("\nNo pending datasets. Nothing to run.")
        return

    if args.dry_run:
        print("\nDry run. These datasets would be executed:")
        for dataset_name in pending:
            print(f"  [RUN]  {dataset_name}")
        return

    print("\nStarting benchmark execution...\n")

    completed_results = []

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:

        future_map = {
            executor.submit(benchmark_dataset, dataset_name): dataset_name
            for dataset_name in pending
        }

        for future in concurrent.futures.as_completed(future_map):
            dataset_name = future_map[future]

            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "dataset": dataset_name,
                    "return_code": -1,
                    "elapsed_seconds": 0,
                    "output": str(exc),
                    "success": False,
                }

            completed_results.append(result)

            if result["success"]:
                print(
                    f"[PASS] {dataset_name} "
                    f"({result['elapsed_seconds']:.1f}s)"
                )
            else:
                print(
                    f"[FAIL] {dataset_name} "
                    f"({result['elapsed_seconds']:.1f}s)"
                )

                print("-" * 75)
                print(result["output"][-4000:])
                print("-" * 75)

    log_rows = []

    for result in completed_results:
        log_rows.append(
            {
                "dataset": result["dataset"],
                "success": result["success"],
                "return_code": result["return_code"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )

    log_df = pd.DataFrame(log_rows)

    log_path = RESULTS_DIR / "batch_benchmark_log.csv"
    log_df.to_csv(log_path, index=False)

    print("\n" + "=" * 75)
    print("BATCH BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"Completed this run : {len(completed_results)}")
    print(
        f"Successful         : "
        f"{sum(bool(x['success']) for x in completed_results)}"
    )
    print(
        f"Failed             : "
        f"{sum(not bool(x['success']) for x in completed_results)}"
    )
    print(f"Batch log          : {log_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
