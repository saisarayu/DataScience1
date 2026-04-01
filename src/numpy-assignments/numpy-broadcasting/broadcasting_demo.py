import numpy as np

# -------------------------------
# 1. Scalar to Array Broadcasting
# -------------------------------
arr = np.array([1, 2, 3])
print("Array:", arr)
print("Shape of arr:", arr.shape)

result_scalar = arr + 10
print("\nScalar Broadcasting (arr + 10):", result_scalar)


# -------------------------------
# 2. 1D to 2D Broadcasting
# -------------------------------
arr_2d = np.array([[1, 2, 3],
                   [4, 5, 6]])

arr_1d = np.array([10, 20, 30])

print("\n2D Array:\n", arr_2d)
print("Shape of 2D array:", arr_2d.shape)

print("\n1D Array:", arr_1d)
print("Shape of 1D array:", arr_1d.shape)

# Broadcasting happens here
result_broadcast = arr_2d + arr_1d

print("\nBroadcasting Result (2D + 1D):\n", result_broadcast)


# -------------------------------
# 3. Example of incompatible shapes
# -------------------------------
arr_wrong = np.array([1, 2])  # shape (2,)

print("\nWrong Array:", arr_wrong)
print("Shape of wrong array:", arr_wrong.shape)

# Uncomment to see error
# print(arr_2d + arr_wrong)