# Numeric Data Types
a = 10          # integer
b = 5.5         # float

print("Integer:", a)
print("Float:", b)

# Arithmetic operations
print("Addition:", a + b)
print("Multiplication:", a * b)

# String Data Types
name = "Sarayu"
message = "Hello"

print("Name:", name)
print("Message:", message)

# String concatenation
print(message + " " + name)

# Type checking
print("Type of a:", type(a))
print("Type of name:", type(name))

# Type mismatch example
# print("Result:", "10" + 5)  # This will cause error

# Fix using conversion
print("Fixed Result:", int("10") + 5)