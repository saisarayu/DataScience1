# 1. Import library
import pandas as pd

# 2. Load dataset
df = pd.read_csv("data/dataset.csv")

# 3. Preview data
print("=== HEAD ===")
print(df.head())

# 4. Inspect structure
print("\n=== INFO ===")
df.info()

# 5. Summary statistics
print("\n=== DESCRIBE ===")
print(df.describe())