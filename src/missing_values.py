import pandas as pd

# 1. Load dataset
df = pd.read_csv("data/dataset.csv")

# 2. Detect missing values (boolean mask)
print("=== MISSING VALUES (TRUE/FALSE) ===")
print(df.isnull())

# 3. Count missing values per column
print("\n=== MISSING COUNT PER COLUMN ===")
print(df.isnull().sum())

# 4. Columns with missing data only
print("\n=== COLUMNS WITH MISSING VALUES ===")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 5. Rows with missing values
print("\n=== ROWS WITH MISSING VALUES ===")
print(df[df.isnull().any(axis=1)])