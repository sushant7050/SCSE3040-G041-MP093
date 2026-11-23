
import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/train.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("\n=== 1. UNIT MIX-UP ===")

print("\nLarge stock values by block:")
print(
    df[df["stock_kg_morning"] > 10000]
    .groupby("block")["stock_kg_morning"]
    .agg(["count", "min", "median", "max"])
)

print("\nStock statistics by block:")
print(
    df.groupby("block")["stock_kg_morning"]
    .agg(["count", "median", "max"])
)

print("\n=== 2. TARGET TYPOS ===")

ratio = (
    df["grain_kg"] /
    df["kg_entered_in_epos_at_close"].replace(0, np.nan)
)

print("Target / EPOS ratio:")
print(ratio.describe())

print("\nRatios approximately 10:")
print(
    df.loc[
        ratio.between(9.5, 10.5),
        ["record_id", "grain_kg",
         "kg_entered_in_epos_at_close"]
    ].to_string(index=False)
)

print("\n=== 3. SPELLING VARIANTS ===")

for col in ["shop_area", "block"]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False).to_string())

    print("\nNormalized categories:")
    print(
        df[col].astype(str).str.strip().str.lower()
        .value_counts().to_string()
    )

print("\n=== 4. STUCK SENSOR INVESTIGATION ===")

numeric_columns = [
    "cards_attached",
    "working_hours",
    "biometric_failures",
    "temp_c",
    "stock_kg_morning",
    "days_to_month_end",
    "kg_entered_in_epos_at_close",
    "grain_kg"
]

for col in numeric_columns:
    daily = df.groupby("date")[col].nunique().sort_index()

    constant_dates = daily[daily == 1].index

    print(
        f"{col}: dates with only one unique value = "
        f"{len(constant_dates)}"
    )

print("\n=== AUDIT COMPLETE ===")
