
import pandas as pd
import numpy as np


def clean_data(df):
    df = df.copy()

    # 1. Remove exact duplicate rows
    df = df.drop_duplicates().copy()

    # 2. Correct sentinel values
    df["cards_attached"] = df["cards_attached"].replace(-1, np.nan)

    # 3. Correct unit mix-up in Block South
    south_mask = (
        df["block"].astype(str).str.strip().str.lower()
        == "block south"
    )

    df.loc[south_mask, "stock_kg_morning"] /= 1000

    # 4. Normalize categorical text
    for col in ["shop_area", "block"]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].str.title()

    # 5. Parse dates
    df["date"] = pd.to_datetime(df["date"])

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/raw/train.csv")

    cleaned = clean_data(df)

    print("Original rows:", len(df))
    print("Cleaned rows:", len(cleaned))
    print("Remaining duplicates:", cleaned.duplicated().sum())
    print("Missing cards:", cleaned["cards_attached"].isna().sum())
    print("Block South stock maximum:",
          cleaned.loc[
              cleaned["block"] == "Block South",
              "stock_kg_morning"
          ].max())
