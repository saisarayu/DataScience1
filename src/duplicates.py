import pandas as pd

# 1. Load dataset
df = pd.read_csv("data/dataset.csv")

# 2. Original shape
print("Original Shape:", df.shape)

# 3. Detect duplicates
duplicates = df.duplicated()

print("\n=== DUPLICATE ROWS (TRUE/FALSE) ===")
print(duplicates)

# 4. Count duplicates
print("\nTotal duplicate rows:", duplicates.sum())

# 5. Inspect duplicate rows
print("\n=== DUPLICATE ENTRIES ===")
print(df[duplicates])

# 6. Remove duplicates
df_cleaned = df.drop_duplicates()

print("\n=== AFTER REMOVING DUPLICATES ===")
print("New Shape:", df_cleaned.shape)

# 7. Verify removal
print("\nDuplicates after cleaning:", df_cleaned.duplicated().sum())