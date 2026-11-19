
import pandas as pd

df = pd.read_csv("data/raw/train.csv")

# 1. Inspect duplicate record IDs
print("\n--- DUPLICATE RECORD IDs ---")

duplicate_ids = df[df["record_id"].duplicated(keep=False)]

print("Rows with duplicate IDs:", len(duplicate_ids))
print("Number of duplicated IDs:", duplicate_ids["record_id"].nunique())

print(duplicate_ids.sort_values("record_id").to_string(index=False))


# 2. Inspect exact duplicate rows
print("\n--- EXACT DUPLICATE ROWS ---")

print(df[df.duplicated(keep=False)].sort_values("record_id").to_string(index=False))


# 3. Investigate stock outliers
print("\n--- STOCK ANALYSIS ---")

print(df["stock_kg_morning"].describe())

print("\nLargest 10 stock values:")
print(
    df.nlargest(10, "stock_kg_morning")[
        ["record_id", "date", "shop_area", "block", "stock_kg_morning"]
    ].to_string(index=False)
)

Q1 = df["stock_kg_morning"].quantile(0.25)
Q3 = df["stock_kg_morning"].quantile(0.75)
IQR = Q3 - Q1

upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR

print("\nIQR lower bound:", lower_bound)
print("IQR upper bound:", upper_bound)

print("Potential IQR outliers:",
      ((df["stock_kg_morning"] < lower_bound) |
       (df["stock_kg_morning"] > upper_bound)).sum())


# 4. Check date validity
print("\n--- DATE VALIDATION ---")

parsed_dates = pd.to_datetime(df["date"], errors="coerce")

print("Invalid dates:", parsed_dates.isna().sum())
print("Minimum date:", parsed_dates.min())
print("Maximum date:", parsed_dates.max())


# 5. Inspect categorical values
print("\n--- CATEGORICAL VALUES ---")

for col in ["shop_area", "block"]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))


# 6. Numeric ranges
print("\n--- NUMERIC RANGES ---")

numeric_cols = df.select_dtypes(include="number").columns

print(df[numeric_cols].agg(["min", "max"]).T)
print("\n--- SUMMARY ---")

print("Duplicate IDs:", df["record_id"].duplicated().sum())

print("Exact duplicate rows:", df.duplicated().sum())

print("Missing values:", df.isnull().sum().sum())

print("Stock values above 10000:",
      (df["stock_kg_morning"] > 10000).sum())

print("Invalid dates:",
      pd.to_datetime(df["date"], errors="coerce").isna().sum())

print("\n--- TOP 5 STOCK VALUES ---")
print(df.nlargest(5, "stock_kg_morning")[
    ["record_id", "date", "stock_kg_morning"]
].to_string(index=False))

print("\n--- NEGATIVE CARDS ---")

print(df[df["cards_attached"] < 0][
    ["record_id", "date", "cards_attached", "block"]
].to_string(index=False))


print("\n--- STOCK VALUES ABOVE 10000 ---")

print("Count:",
      (df["stock_kg_morning"] > 10000).sum())

print(df[df["stock_kg_morning"] > 10000][
    ["record_id", "date", "stock_kg_morning", "shop_area", "block"]
].head(10).to_string(index=False))

stock = df["stock_kg_morning"]

print("\n--- FINAL STOCK INVESTIGATION ---")

print("Negative cards:",
      (df["cards_attached"] < 0).sum())

print("Stock above 10000:",
      (stock > 10000).sum())

print("Stock above 10000 divisible by 1000:",
      ((stock > 10000) & (stock % 1000 == 0)).sum())

print("Stock between 0 and 10000:",
      ((stock > 0) & (stock <= 10000)).sum())

print("\nStock distribution:")
print(stock.describe())
