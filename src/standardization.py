import pandas as pd

data = {
    "User Name": ["Alice ", "BOB", "charlie"],
    "AGE ": ["25", "30", "35"],
    "Join-Date": ["2024/01/01", "01-02-2024", "March 3, 2024"]
}

df = pd.DataFrame(data)
print("Before:\n", df)