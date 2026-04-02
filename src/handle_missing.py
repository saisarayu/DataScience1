import pandas as pd

# 1. Load dataset
df = pd.read_csv("data/dataset.csv")

# 2. Check missing values BEFORE handling
print("=== MISSING VALUES BEFORE ===")
print(df.isnull().sum())

# 3. Store original shape
print("\nOriginal Shape:", df.shape)

# -------------------------------
# 4. DROP STRATEGY
# -------------------------------

# Drop rows with missing values
df_dropped = df.dropna()

print("\n=== AFTER DROPPING ROWS ===")
print(df_dropped.shape)

# -------------------------------
# 5. FILL STRATEGY
# -------------------------------

df_filled = df.copy()

# Fill numeric columns with mean
df_filled = df_filled.fillna(df_filled.mean(numeric_only=True))

# Fill categorical columns with mode
for col in df_filled.select_dtypes(include='object').columns:
    df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)

print("\n=== AFTER FILLING VALUES ===")
print(df_filled.isnull().sum())
print("Shape after filling:", df_filled.shape)