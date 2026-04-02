import pandas as pd

# Load dataset
df = pd.read_csv("data/dataset.csv")

# 1. Select single column
print("=== SINGLE COLUMN ===")
print(df['column_name'])

# 2. Select multiple columns
print("\n=== MULTIPLE COLUMNS ===")
print(df[['column1', 'column2']])

# 3. Select rows using position (iloc)
print("\n=== ROWS USING ILOC ===")
print(df.iloc[0])           # first row
print(df.iloc[0:5])         # first 5 rows

# 4. Select rows using labels (loc)
print("\n=== ROWS USING LOC ===")
print(df.loc[0])            # row with index label 0
print(df.loc[0:5])          # rows from label 0 to 5

# 5. Combined row + column selection
print("\n=== COMBINED SELECTION ===")
print(df.loc[0:5, ['column1', 'column2']])