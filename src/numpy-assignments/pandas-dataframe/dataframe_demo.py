import pandas as pd

# -------------------------------
# 1. Create DataFrame from Dictionary
# -------------------------------
data = {
    "Name": ["Sarayu", "Rahul", "Anita"],
    "Age": [21, 22, 20],
    "City": ["Hosur", "Bangalore", "Chennai"]
}

df_dict = pd.DataFrame(data)

print("DataFrame from Dictionary:")
print(df_dict)

print("\nColumns:", df_dict.columns)
print("Shape:", df_dict.shape)


# -------------------------------
# 2. Load DataFrame from CSV File
# -------------------------------
df_file = pd.read_csv("sample_data.csv")

print("\nDataFrame from CSV:")
print(df_file)

print("\nColumns:", df_file.columns)
print("Shape:", df_file.shape)


# -------------------------------
# 3. Inspect First Few Rows
# -------------------------------
print("\nFirst 2 rows:")
print(df_file.head(2))