# Function with return
def add_numbers(a, b):
    result = a + b
    return result


# Call function and store result
sum_result = add_numbers(10, 20)

# Use returned value
print("Sum:", sum_result)


# Use returned value in another calculation
double_result = sum_result * 2
print("Double of sum:", double_result)