from pathlib import Path
import subprocess
import sys


DATASETS = [
    "pima",
    "glass1",
    "wisconsin",
]


def run_dataset(dataset):
    print("=" * 70)
    print(f"Running dataset: {dataset}")
    print("=" * 70)

    command = [
        sys.executable,
        "src/run_experiment.py",
        "--dataset",
        dataset
    ]

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"FAILED: {dataset}")
        return False

    print(f"COMPLETED: {dataset}")
    return True


def main():
    print("=" * 70)
    print("ResampleRank - Dataset Experiment Runner")
    print("=" * 70)

    for dataset in DATASETS:
        run_dataset(dataset)

    print("\nAll requested datasets processed.")


if __name__ == "__main__":
    main()