import numpy as np

# 1D Array
arr1 = np.array([10, 20, 30, 40])
print("1D Array:", arr1)
print("Shape:", arr1.shape)
print("Dimensions:", arr1.ndim)

# Access elements
print("First element:", arr1[0])
print("Last element:", arr1[3])

print("\n------------------\n")

# 2D Array
arr2 = np.array([[1, 2, 3],
                 [4, 5, 6]])

print("2D Array:\n", arr2)
print("Shape:", arr2.shape)
print("Dimensions:", arr2.ndim)

# Access elements
print("Element at (0,1):", arr2[0,1])
print("Element at (1,2):", arr2[1,2])