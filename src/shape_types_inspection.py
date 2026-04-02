# 1. Import
import pandas as pd

# 2. Load dataset
df = pd.read_csv("data/dataset.csv")

# 3. Inspect shape
print("=== SHAPE ===")
print(df.shape)

# 4. Interpret shape
rows, cols = df.shape
print(f"Rows: {rows}")
print(f"Columns: {cols}")

# 5. Inspect data types
print("\n=== DATA TYPES ===")
print(df.dtypes)