#!/usr/bin/env python
# coding: utf-8

# In[1]:


# question 1

# Create a tuple of colors
colors = ("Red", "Blue", "Green", "Yellow", "Blue", "Green")

# Access elements using indexing
print("Accessing elements using indexing:")
print(f"First color: {colors[0]}")
print(f"Last color: {colors[-1]}")

# Try to modify an element in the tuple (this will raise an error)
try:
    colors[1] = "Purple"
except TypeError as e:
    print("\nAttempt to modify an element:")
    print(f"Error: {e}")

# Find the number of occurrences of a specific element in the tuple
color_to_count = "Blue"
occurrences = colors.count(color_to_count)
print(f"\nNumber of occurrences of '{color_to_count}' in the tuple: {occurrences}")


# In[3]:


# question 2

# Create a dictionary to store information about a person
person = {
    "name": "Asmi Garg",
    "age": 21,
    "city": "Dehradun"
}

# Access values using keys
print("Accessing values using keys:")
print(f"Name: {person['name']}")
print(f"Age: {person['age']}")
print(f"City: {person['city']}")

# Add a new key-value pair to the dictionary
person["email"] = "asmi.garg@example.com"
print("\nAfter adding a new key-value pair:")
print(person)

# Modify an existing value
person["age"] = 31
print("\nAfter modifying the age:")
print(person)

# Check if a key exists in the dictionary
key_to_check = "city"
if key_to_check in person:
    print(f"\nThe key '{key_to_check}' exists in the dictionary.")
else:
    print(f"\nThe key '{key_to_check}' does not exist in the dictionary.")

# Get a list of all keys and values
keys = list(person.keys())
values = list(person.values())
print("\nList of all keys:", keys)
print("List of all values:", values)


# In[4]:


#question 3

# Create a list (mutable data type)
my_list = [10, 20, 30]

# Create a tuple (immutable data type)
my_tuple = (10, 20, 30)

# Try to modify an element in the list
print("Original list:", my_list)
my_list[1] = 99
print("Modified list:", my_list)

# Try to modify an element in the tuple (this will raise an error)
print("\nOriginal tuple:", my_tuple)
try:
    my_tuple[1] = 99
except TypeError as e:
    print("Attempt to modify tuple:")
    print(f"Error: {e}")


# In[5]:


# question 4

# Take a number as input from the user
number = int(input("Enter a number: "))

# Use a for loop to iterate from 1 to 10
print(f"\nMultiplication Table of {number}:")
for i in range(1, 11):
    # Calculate the product of the input number and the current iteration
    product = number * i
    # Print the multiplication table
    print(f"{number} x {i} = {product}")


# In[6]:


# question 5

# Function to calculate the factorial of a number
def factorial(n):
    if n < 0:
        return "Factorial is not defined for negative numbers."
    elif n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

# Take a number as input from the user
number = int(input("Enter a number to find its factorial: "))

# Calculate the factorial and print the result
fact = factorial(number)
print(f"The factorial of {number} is: {fact}")


# In[7]:


# question 6

def is_prime(n):
    if n <= 1:
        return False
    elif n == 2:
        return True
    elif n % 2 == 0:
        return False
    else:
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True

number = int(input("Enter a number to check if it is prime: "))

if is_prime(number):
    print(f"{number} is a prime number.")
else:
    print(f"{number} is not a prime number.")


# In[8]:


sum_even = 0

for number in range(1, 101):
    if number % 2 == 0:
        sum_even += number

print(f"The sum of all even numbers between 1 and 100 is: {sum_even}")


# In[ ]:


# question 8

def calculator():
    while True:
        print("\nSimple Calculator")
        print("Choose an operation:")
        print("1. Addition")
        print("2. Subtraction")
        print("3. Multiplication")
        print("4. Division")
        print("5. Exit")
        
        choice = input("Enter choice (1/2/3/4/5): ")
        
        if choice == '5':
            print("Exiting the calculator. Goodbye!")
            break
        
        if choice in ['1', '2', '3', '4']:
            num1 = float(input("Enter first number: "))
            num2 = float(input("Enter second number: "))
            
            if choice == '1':
                result = num1 + num2
                print(f"The result of addition is: {result}")
            elif choice == '2':
                result = num1 - num2
                print(f"The result of subtraction is: {result}")
            elif choice == '3':
                result = num1 * num2
                print(f"The result of multiplication is: {result}")
            elif choice == '4':
                if num2 != 0:
                    result = num1 / num2
                    print(f"The result of division is: {result}")
                else:
                    print("Error: Division by zero is not allowed.")
        else:
            print("Invalid input. Please choose a valid operation.")

calculator()


# In[ ]:




