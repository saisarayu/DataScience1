import pandas as pd
import numpy as np

# -------------------------------
# 1. Create Series from List
# -------------------------------
data_list = [10, 20, 30, 40]

series_from_list = pd.Series(data_list)

print("Series from List:")
print(series_from_list)

print("\nValues:", series_from_list.values)
print("Index:", series_from_list.index)


# -------------------------------
# 2. Create Series from NumPy Array
# -------------------------------
data_array = np.array([1, 2, 3, 4])

series_from_array = pd.Series(data_array)

print("\nSeries from NumPy Array:")
print(series_from_array)

print("\nValues:", series_from_array.values)
print("Index:", series_from_array.index)