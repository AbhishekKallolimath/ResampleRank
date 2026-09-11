import pandas as pd
import sys


def profile_dataset(data_path, target_path):
    X = pd.read_csv(data_path)
    y = pd.read_csv(target_path).iloc[:, 0]

    class_counts = y.value_counts()

    majority_count = class_counts.max()
    minority_count = class_counts.min()

    imbalance_ratio = majority_count / minority_count

    profile = {
        "dataset": data_path.replace("\\", "/").split("/")[-1].replace(".csv", ""),
        "samples": X.shape[0],
        "features": X.shape[1],
        "feature_sample_ratio": round(X.shape[1] / X.shape[0], 4),
        "numeric_features": int(
            X.select_dtypes(include="number").shape[1]
        ),
        "categorical_features": int(
            X.select_dtypes(exclude="number").shape[1]
        ),
        "missing_values": int(X.isnull().sum().sum()),
        "duplicates": int(X.duplicated().sum()),
        "majority_class": class_counts.idxmax(),
        "minority_class": class_counts.idxmin(),
        "majority_count": int(majority_count),
        "minority_count": int(minority_count),
        "minority_proportion": round(minority_count / len(y), 3),
        "imbalance_ratio": round(imbalance_ratio, 3),
    }

    return profile


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python src\\dataset_profiler.py "
            "<features.csv> <target.csv>"
        )
        sys.exit(1)

    data_path = sys.argv[1]
    target_path = sys.argv[2]

    result = profile_dataset(data_path, target_path)

    print("Dataset Profile")
    print("----------------")

    for key, value in result.items():
        print(f"{key}: {value}")