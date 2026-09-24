from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    BASE_DIR
    / "results"
)

GROUP_RESULTS_DIR = (
    RESULTS_DIR
    / "sensitivity"
    / "vowel0_group"
)

CLASSIFIERS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
]


def calculate_f1(
    tp,
    fp,
    fn,
):
    precision_denominator = (
        tp + fp
    )

    recall_denominator = (
        tp + fn
    )

    precision = (
        tp / precision_denominator
        if precision_denominator != 0
        else 0.0
    )

    recall = (
        tp / recall_denominator
        if recall_denominator != 0
        else 0.0
    )

    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
    )


def main():

    print(
        "=" * 75
    )

    print(
        "ResampleRank - Updating Vowel0 Primary Results"
    )

    print(
        "=" * 75
    )

    for classifier in CLASSIFIERS:

        original_file = (
            RESULTS_DIR
            / f"vowel0_{classifier}_results.csv"
        )

        group_file = (
            GROUP_RESULTS_DIR
            / f"vowel0_{classifier}_group_results.csv"
        )

        if not original_file.exists():
            raise FileNotFoundError(
                f"Original result file not found: "
                f"{original_file}"
            )

        if not group_file.exists():
            raise FileNotFoundError(
                f"Group result file not found: "
                f"{group_file}"
            )

        original_df = pd.read_csv(
            original_file
        )

        group_df = pd.read_csv(
            group_file
        )

        if len(original_df) != len(group_df):
            raise ValueError(
                f"Row count mismatch for "
                f"{classifier}: "
                f"{len(original_df)} vs "
                f"{len(group_df)}"
            )

        original_df = (
            original_df
            .set_index(
                "resampling_code"
            )
        )

        group_df = (
            group_df
            .set_index(
                "resampling_code"
            )
        )

        required_codes = set(
            group_df.index
        )

        if set(original_df.index) != required_codes:
            raise ValueError(
                f"Resampling strategy mismatch "
                f"for {classifier}"
            )

        for code in group_df.index:

            row = group_df.loc[
                code
            ]

            tn = float(
                row["tn"]
            )

            fp = float(
                row["fp"]
            )

            fn = float(
                row["fn"]
            )

            tp = float(
                row["tp"]
            )

            f1_class_0 = calculate_f1(
                tn,
                fn,
                fp,
            )

            f1_class_1 = calculate_f1(
                tp,
                fp,
                fn,
            )

            macro_f1 = (
                f1_class_0
                + f1_class_1
            ) / 2.0

            original_df.loc[
                code,
                "tn"
            ] = tn

            original_df.loc[
                code,
                "fp"
            ] = fp

            original_df.loc[
                code,
                "fn"
            ] = fn

            original_df.loc[
                code,
                "tp"
            ] = tp

            original_df.loc[
                code,
                "accuracy"
            ] = row[
                "accuracy"
            ]

            original_df.loc[
                code,
                "balanced_accuracy"
            ] = row[
                "balanced_accuracy"
            ]

            original_df.loc[
                code,
                "precision"
            ] = row[
                "precision"
            ]

            original_df.loc[
                code,
                "recall"
            ] = row[
                "recall"
            ]

            original_df.loc[
                code,
                "f1_class_0_math"
            ] = f1_class_0

            original_df.loc[
                code,
                "f1_class_1_math"
            ] = f1_class_1

            original_df.loc[
                code,
                "macro_f1_code"
            ] = row[
                "macro_f1_code"
            ]

            original_df.loc[
                code,
                "macro_f1_math"
            ] = macro_f1

            original_df.loc[
                code,
                "metric_error"
            ] = row[
                "metric_error"
            ]

            original_df.loc[
                code,
                "verification"
            ] = row[
                "verification"
            ]

            original_df.loc[
                code,
                "execution_time_seconds"
            ] = row[
                "execution_time_seconds"
            ]

        updated_df = (
            original_df
            .reset_index()
            .sort_values(
                by="macro_f1_math",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        updated_df.to_csv(
            original_file,
            index=False,
        )

        print(
            f"Updated: {original_file}"
        )

    print(
        "\nVowel0 primary result files "
        "successfully updated."
    )


if __name__ == "__main__":
    main()