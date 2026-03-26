# ----------------------
# IMPORTS (if needed)
# ----------------------

# (No imports needed for this example)


# ----------------------
# SETUP / DATA
# ----------------------

numbers = [10, 20, 30, 40]


# ----------------------
# FUNCTIONS (LOGIC)
# ----------------------

def calculate_sum(data):
    """Returns the total of numbers"""
    return sum(data)


def calculate_average(data):
    """Returns the average of numbers"""
    return sum(data) / len(data)


# ----------------------
# MAIN EXECUTION
# ----------------------

def main():
    total = calculate_sum(numbers)
    average = calculate_average(numbers)

    print("Total:", total)
    print("Average:", average)


# ----------------------
# ENTRY POINT
# ----------------------

if __name__ == "__main__":
    main()