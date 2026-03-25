# FOR LOOP (range)
print("For loop with range:")
for i in range(1, 6):
    print(i)


# FOR LOOP (list)
numbers = [10, 20, 30]
print("For loop with list:")
for num in numbers:
    print(num)


# WHILE LOOP
print("While loop:")
count = 1
while count <= 5:
    print(count)
    count += 1


# BREAK example
print("Break example:")
for i in range(1, 10):
    if i == 5:
        break
    print(i)


# CONTINUE example
print("Continue example:")
for i in range(1, 6):
    if i == 3:
        continue
    print(i)