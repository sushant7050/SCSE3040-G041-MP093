import pandas as pd

from mp093.cleaning import clean_data
from mp093.monitoring import calculate_psi
from mp093.config import ROOT


def main():
    train = pd.read_csv(ROOT / "data" / "raw" / "train.csv")
    later = pd.read_csv(ROOT / "data" / "raw" / "later.csv")

    # Apply the project's existing cleaning process
    train_clean = clean_data(train)
    later_clean = clean_data(later)

    numeric_features = [
        "cards_attached",
        "working_hours",
        "biometric_failures",
        "temp_c",
        "stock_kg_morning",
        "days_to_month_end",
    ]

    print("\n--- PSI: Cleaned Train vs Later Data ---\n")

    results = {}

    for feature in numeric_features:
        psi = calculate_psi(
            train_clean[feature],
            later_clean[feature],
            bins=10
        )

        results[feature] = psi
        print(f"{feature}: {psi:.4f}")

    most_drifted = max(results, key=results.get)

    print("\nMost drifted feature:", most_drifted)
    print("PSI:", round(results[most_drifted], 4))


if __name__ == "__main__":
    main()