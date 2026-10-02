import glob
import pandas as pd


def main():

    # Find all final result files
    result_files = glob.glob(
        "results/*_results.csv"
    )

    all_results = []

    for file in result_files:

        result = pd.read_csv(file)

        all_results.append(result)

    # Combine all 12 experiments
    results = pd.concat(
        all_results,
        ignore_index=True
    )

    # Convert accuracy from 0-1 to percentage
    results["accuracy_percent"] = (
        results["accuracy"] * 100
    )

    print("\nIndividual experiment results:\n")

    print(
        results[
            [
                "augmentation",
                "seed",
                "accuracy_percent"
            ]
        ].sort_values(
            ["augmentation", "seed"]
        )
    )

    # Calculate statistics for each augmentation
    summary = (
        results
        .groupby("augmentation")[
            "accuracy_percent"
        ]
        .agg(
            mean_accuracy="mean",
            std_accuracy="std",
            runs="count"
        )
        .reset_index()
    )

    summary = summary.sort_values(
        "mean_accuracy",
        ascending=False
    )

    print("\nSummary:\n")

    print(summary)

    # Save summary
    summary.to_csv(
        "results/summary.csv",
        index=False
    )

    print(
        "\nSummary saved to "
        "results/summary.csv"
    )


if __name__ == "__main__":
    main()