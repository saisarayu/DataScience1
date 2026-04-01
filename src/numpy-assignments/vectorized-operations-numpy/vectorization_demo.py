import numpy as np

# -------------------------------
# 1. Create NumPy Array
# -------------------------------
arr = np.array([1, 2, 3, 4, 5])
print("Original Array:", arr)


# -------------------------------
# 2. Loop-Based Approach
# -------------------------------
result_loop = []

for num in arr:
    result_loop.append(num ** 2)

result_loop = np.array(result_loop)
print("\nLoop-Based Result:", result_loop)


# -------------------------------
# 3. Vectorized Approach
# -------------------------------
result_vectorized = arr ** 2
print("Vectorized Result:", result_vectorized)


# -------------------------------
# 4. Verify Both Results
# -------------------------------
are_equal = np.array_equal(result_loop, result_vectorized)
print("\nAre both results equal?", are_equal)