from dataset_profiler import profile_dataset
import pandas as pd

datasets = [
    ("pima", "data/raw/pima.csv", "data/raw/pima_target.csv"),
    ("glass1", "data/raw/glass1.csv", "data/raw/glass1_target.csv"),
    ("wisconsin", "data/raw/wisconsin.csv", "data/raw/wisconsin_target.csv"),

    ("yeast1", "data/raw/yeast1.csv", "data/raw/yeast1_target.csv"),
    ("haberman", "data/raw/haberman.csv", "data/raw/haberman_target.csv"),
    ("vehicle2", "data/raw/vehicle2.csv", "data/raw/vehicle2_target.csv"),
    ("iris0", "data/raw/iris0.csv", "data/raw/iris0_target.csv"),
    ("segment0", "data/raw/segment0.csv", "data/raw/segment0_target.csv"),
    ("page-blocks0", "data/raw/page-blocks0.csv", "data/raw/page-blocks0_target.csv"),
    ("vowel0", "data/raw/vowel0.csv", "data/raw/vowel0_target.csv"),
    ("shuttle-c0-vs-c4", "data/raw/shuttle-c0-vs-c4.csv", "data/raw/shuttle-c0-vs-c4_target.csv"),
    ("ecoli-0_vs_1", "data/raw/ecoli-0_vs_1.csv", "data/raw/ecoli-0_vs_1_target.csv"),
    ("zoo-3", "data/raw/zoo-3.csv", "data/raw/zoo-3_target.csv"),
    ("abalone9-18", "data/raw/abalone9-18.csv", "data/raw/abalone9-18_target.csv"),
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