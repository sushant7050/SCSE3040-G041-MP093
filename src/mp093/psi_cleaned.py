"""PSI of every numeric column: cleaned train.csv vs later.csv (PROBLEM.md 7.1).

later.csv is already clean (PROBLEM.md section 3), so it is used as given.
Cleaning it again would divide Block South's stock_kg_morning, which later.csv
already records in kg, by 1000 a second time.

Run from anywhere (after pip install -e .):
    python -m mp093.psi_cleaned
"""

from mp093.config import load_params
from mp093.data import clean, load_raw
from mp093.monitoring import calculate_psi


def main():
    params = load_params()

    train_clean = clean(load_raw(params, "train"), params)
    later = load_raw(params, "later")   # already clean: clean() must NOT be run on it

    # The numeric request fields: every feature that is not a text column.
    numeric_features = [
        f for f in params["features"] if f not in params["train"]["one_hot"]
    ]

    print("\n--- PSI: Cleaned Train vs Later Data ---\n")

    results = {}

    for feature in numeric_features:
        psi = calculate_psi(
            train_clean[feature],
            later[feature],
            bins=params["monitoring"]["psi_bins"]
        )

        results[feature] = psi
        print(f"{feature}: {psi:.4f}")

    most_drifted = max(results, key=results.get)

    print("\nMost drifted feature:", most_drifted)
    print("PSI:", round(results[most_drifted], 4))


if __name__ == "__main__":
    main()
