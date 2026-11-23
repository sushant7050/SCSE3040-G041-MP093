
import pandas as pd

df = pd.read_csv("data/raw/train.csv")
df["date"] = pd.to_datetime(df["date"])

columns = [
    "cards_attached",
    "working_hours",
    "biometric_failures",
    "temp_c",
    "stock_kg_morning",
]

for col in columns:
    print(f"\n{'='*60}")
    print("COLUMN:", col)
    print("="*60)

    # Count repeated values across different dates
    repeated = (
        df.groupby(col)
        .agg(
            count=("date", "size"),
            unique_dates=("date", "nunique"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .reset_index()
    )

    repeated = repeated[
        (repeated["count"] >= 10) &
        (repeated["unique_dates"] >= 5)
    ]

    print(
        repeated.sort_values(
            "unique_dates", ascending=False
        ).head(10).to_string(index=False)
    )

print("\nFINAL AUDIT COMPLETE")
