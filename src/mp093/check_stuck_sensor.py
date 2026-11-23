
import pandas as pd

df = pd.read_csv("data/raw/train.csv")
df["date"] = pd.to_datetime(df["date"])

columns = [
    "cards_attached",
    "working_hours",
    "biometric_failures",
    "temp_c",
    "stock_kg_morning"
]

for block in df["block"].unique():

    print(f"\n{'='*60}")
    print("BLOCK:", block)
    print("="*60)

    block_df = df[df["block"] == block]

    for col in columns:

        daily = (
            block_df.groupby("date")[col]
            .agg(["median", "nunique", "count"])
            .sort_index()
        )

        daily["previous"] = daily["median"].shift(1)

        daily["same_as_previous"] = (
            daily["median"] == daily["previous"]
        )

        print(f"\nColumn: {col}")

        print(
            daily[daily["same_as_previous"]]
            .head(15)
            .to_string()
        )

print("\nBLOCK SENSOR INVESTIGATION COMPLETE")
