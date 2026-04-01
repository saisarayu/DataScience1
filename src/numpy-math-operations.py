import numpy as np

# 1. Create NumPy arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

print("Array 1:", arr1)
print("Array 2:", arr2)

# 2. Element-wise operations
addition = arr1 + arr2
subtraction = arr1 - arr2
multiplication = arr1 * arr2
division = arr1 / arr2

print("\nElement-wise Addition:", addition)
print("Element-wise Subtraction:", subtraction)
print("Element-wise Multiplication:", multiplication)
print("Element-wise Division:", division)

# 3. Scalar operation
scalar_add = arr1 + 10
scalar_multiply = arr1 * 2

print("\nScalar Addition (arr1 + 10):", scalar_add)
print("Scalar Multiplication (arr1 * 2):", scalar_multiply)

# 4. Shape demonstration
print("\nShape of arr1:", arr1.shape)
print("Shape of arr2:", arr2.shape)

# 5. Example of incompatible shapes (for explanation purpose)
arr3 = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2,3)

print("\nArray 3:", arr3)
print("Shape of arr3:", arr3.shape)

# Uncommenting below will raise error
# print(arr1 + arr3)