import numpy as np
import pandas as pd


def calculate_psi(reference, new_data, bins=10):
    reference = reference.dropna()
    new_data = new_data.dropna()

    edges = np.quantile(
        reference,
        np.linspace(0, 1, bins + 1)
    )

    edges = np.unique(edges)

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(
        reference, bins=edges
    )

    new_counts, _ = np.histogram(
        new_data, bins=edges
    )

    ref_pct = ref_counts / len(reference)
    new_pct = new_counts / len(new_data)

    epsilon = 1e-6

    psi = np.sum(
        (new_pct - ref_pct)
        * np.log(
            (new_pct + epsilon)
            / (ref_pct + epsilon)
        )
    )

    return float(psi)


def should_promote(mae_v1, mae_v2, max_ratio=0.95):
    """The client's rule: promote v2 only if its MAE is at least 5% lower than v1's.

    max_ratio comes from params.yaml (monitoring.promote_max_ratio).
    Exactly 5% lower (mae_v2 == 0.95 * mae_v1) still counts as promote.
    """
    return mae_v2 <= max_ratio * mae_v1


if __name__ == "__main__":
    train = pd.read_csv("data/raw/train.csv")
    later = pd.read_csv("data/raw/later.csv")

    numeric_features = [
        "cards_attached",
        "working_hours",
        "biometric_failures",
        "temp_c",
        "stock_kg_morning",
        "days_to_month_end",
    ]

    for feature in numeric_features:
        psi = calculate_psi(
            train[feature],
            later[feature]
        )
        print(f"{feature}: {psi:.4f}")