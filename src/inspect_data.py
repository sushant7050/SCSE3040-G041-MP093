
import pandas as pd

# Load training data
df = pd.read_csv("data/raw/train.csv")

# 1. Basic information
print("\n--- SHAPE ---")
print(df.shape)

print("\n--- COLUMNS ---")
print(df.columns.tolist())

print("\n--- DATA TYPES ---")
print(df.dtypes)

# 2. Missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# 3. Duplicate rows
print("\n--- DUPLICATES ---")
print(df.duplicated().sum())

# 4. Statistical summary
print("\n--- STATISTICS ---")
print(df.describe(include="all").T)

# 5. First few rows
print("\n--- SAMPLE DATA ---")
print(df.head())

# 6. Unique values
print("\n--- UNIQUE VALUES ---")
print(df.nunique())
