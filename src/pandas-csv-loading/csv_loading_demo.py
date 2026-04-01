import pandas as pd

# -------------------------------
# 1. Load CSV File
# -------------------------------
df = pd.read_csv("sample_data.csv")

print("Data loaded successfully!\n")


# -------------------------------
# 2. Preview Data
# -------------------------------
print("First 5 rows:")
print(df.head())


# -------------------------------
# 3. Inspect Structure
# -------------------------------
print("\nColumns:", df.columns)
print("Shape (rows, columns):", df.shape)


# -------------------------------
# 4. Check Data Types
# -------------------------------
print("\nData Types:")
print(df.dtypes)