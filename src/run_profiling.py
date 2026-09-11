from dataset_profiler import profile_dataset
import pandas as pd


datasets = [
    ("pima", "data/raw/pima.csv", "data/raw/pima_target.csv"),
]


profiles = []

for name, data_path, target_path in datasets:
    print(f"Profiling: {name}")

    profile = profile_dataset(data_path, target_path)
    profiles.append(profile)


profile_df = pd.DataFrame(profiles)

profile_df.to_csv(
    "data/processed/dataset_profiles.csv",
    index=False
)

print("\nAll dataset profiles saved.")
print(profile_df)