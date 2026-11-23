
import pandas as pd

df = pd.read_csv("data/raw/train.csv")
df["date"] = pd.to_datetime(df["date"])

sensor_columns = [
    "cards_attached",
    "working_hours",
    "biometric_failures",
    "temp_c",
    "stock_kg_morning",
]

for block in sorted(df["block"].dropna().unique()):

    block_df = df[df["block"] == block].copy()

    print(f"\n{'=' * 60}")
    print("BLOCK:", block)
    print("=" * 60)

    for col in sensor_columns:

        daily = (
            block_df.groupby("date")[col]
            .agg(["median", "nunique", "count"])
            .reset_index()
            .sort_values("date")
        )

        # Find consecutive identical daily medians
        daily["run"] = daily["median"].ne(
            daily["median"].shift()
        ).cumsum()

        runs = (
            daily.groupby("run")
            .agg(
                value=("median", "first"),
                start_date=("date", "min"),
                end_date=("date", "max"),
                days=("date", "count"),
            )
            .reset_index(drop=True)
        )

        longest = runs.sort_values(
            "days", ascending=False
        ).head(3)

        print(f"\nColumn: {col}")
        print(longest.to_string(index=False))

print("\nBLOCK-LEVEL AUDIT COMPLETE")
