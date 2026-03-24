# LIST (mutable)
numbers = [10, 20, 30]

print("List:", numbers)

# Access element
print("First element:", numbers[0])

# Modify list
numbers.append(40)
print("Updated List:", numbers)


# TUPLE (immutable)
values = (1, 2, 3)

print("Tuple:", values)

# Access element
print("Second element:", values[1])

# Uncommenting below will cause error
# values[0] = 100  # ❌ tuples cannot be changed


# DICTIONARY (key-value pairs)
student = {
    "name": "Sarayu",
    "age": 20,
    "course": "Data Science"
}

print("Dictionary:", student)

# Access value
print("Name:", student["name"])

# Modify dictionary
student["age"] = 21
print("Updated Dictionary:", student)