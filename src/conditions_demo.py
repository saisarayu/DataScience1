# Basic if condition
age = 18

if age >= 18:
    print("You are eligible to vote")


# if-else example
marks = 45

if marks >= 50:
    print("Pass")
else:
    print("Fail")


# if-elif-else example
score = 75

if score >= 90:
    print("Grade A")
elif score >= 70:
    print("Grade B")
elif score >= 50:
    print("Grade C")
else:
    print("Fail")


# Logical operators
temperature = 30
is_sunny = True

if temperature > 25 and is_sunny:
    print("Good weather for outing")

if temperature > 35 or is_sunny:
    print("It might be hot or sunny")

if not is_sunny:
    print("It is not sunny today")