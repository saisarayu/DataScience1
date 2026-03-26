import numpy as np

# ----------------------
# Python List
# ----------------------
list_1d = [1, 2, 3, 4]
list_2d = [[1, 2], [3, 4]]

print("Python List:", list_1d)


# ----------------------
# Convert to NumPy Arrays
# ----------------------
array_1d = np.array(list_1d)
array_2d = np.array(list_2d)

print("1D Array:", array_1d)
print("2D Array:\n", array_2d)


# ----------------------
# Array Properties
# ----------------------
print("Shape of 1D array:", array_1d.shape)
print("Shape of 2D array:", array_2d.shape)

print("Data type:", array_1d.dtype)


# ----------------------
# Basic Operation
# ----------------------
result = array_1d + 10
print("Array after adding 10:", result)