#!/usr/bin/env python
# coding: utf-8

# In[1]:


# question 1

a=float(input("enter the first number:"))
b=float(input("enter the second number:"))

add=a+b
sub=a-b
mul=a*b
div=a/b if b!=0 else "undefined"
mod=a%b if b!=0 else "undefined"

print("\nResults of Arithmetic Operations:")
print(f"Addition: {a} + {b} = {add}")
print(f"Subtraction: {a} - {b} = {sub}")
print(f"Multiplication: {a} * {b} = {mul}")
print(f"Division: {a} / {b} = {div}")
print(f"Modulus: {a} % {b} = {mod}")


# In[2]:


# question 2

integer_var = 200
float_var = 78.7
string_var1 = "UPES"
string_var2 = "Computer Science"
boolean_var1 = True
boolean_var2 = False

addition = integer_var + float_var
subtraction = integer_var - float_var
multiplication = integer_var * float_var
division = integer_var / float_var

concatenated_string = string_var1 + " " + string_var2

and_operation = boolean_var1 and boolean_var2
or_operation = boolean_var1 or boolean_var2
not_operation = not boolean_var1

print("Arithmetic Operations on Numeric Data Types:")
print(f"Addition: {integer_var} + {float_var} = {addition}")
print(f"Subtraction: {integer_var} - {float_var} = {subtraction}")
print(f"Multiplication: {integer_var} * {float_var} = {multiplication}")
print(f"Division: {integer_var} / {float_var} = {division}")

print("\nString Concatenation:")
print(f"Concatenated String: {concatenated_string}")

print("\nLogical Operations on Boolean Data Types:")
print(f"AND Operation: {boolean_var1} and {boolean_var2} = {and_operation}")
print(f"OR Operation: {boolean_var1} or {boolean_var2} = {or_operation}")
print(f"NOT Operation: not {boolean_var1} = {not_operation}")


# In[3]:


# question 3

from datetime import datetime
name=input("Enter your name")
print(f"Hello, {name}!")

birth_year = int(input("Enter your birth year: "))
current_year = datetime.now().year
age = current_year - birth_year

print(f"{name} is {age} years old.")


# In[4]:


# question 4

number = int(input("Enter a number: "))

if number % 2 == 0:
    print(f"The number {number} is even.")
else:
    print(f"The number {number} is odd.")


# In[5]:


# question 5

# Using a for loop to print numbers from 1 to 10
print("Using for loop:")
for i in range(1, 11):
    print(i)

# Using a while loop with a counter variable to print numbers from 1 to 10
print("\nUsing while loop:")
counter = 1
while counter <= 10:
    print(counter)
    counter += 1


# In[6]:


# question 6

# Create a list of fruits
fruits = ["Apple", "Banana", "Cherry", "Date", "Elderberry"]

# Access elements using indexing
print("Accessing elements using indexing:")
print(f"First fruit: {fruits[0]}")
print(f"Last fruit: {fruits[-1]}")

# Modify elements in the list
fruits[1] = "Blueberry"
print("\nAfter modifying the second element:")
print(fruits)

# Add elements to the list
fruits.append("Fig")
print("\nAfter adding an element:")
print(fruits)

# Remove elements from the list
fruits.remove("Date")
print("\nAfter removing an element:")
print(fruits)

# Find the length of the list
length = len(fruits)
print(f"\nLength of the list: {length}")

# Sort the list in ascending order
fruits.sort()
print("\nList sorted in ascending order:")
print(fruits)

# Sort the list in descending order
fruits.sort(reverse=True)
print("\nList sorted in descending order:")
print(fruits)


# In[7]:


# question 7

# Create a string variable
my_string = "Hello, World! Welcome to Python programming."

# Find the length of the string
length = len(my_string)
print(f"Length of the string: {length}")

# Convert the string to uppercase
uppercase_string = my_string.upper()
print(f"String in uppercase: {uppercase_string}")

# Convert the string to lowercase
lowercase_string = my_string.lower()
print(f"String in lowercase: {lowercase_string}")

# Check if a substring exists in the string
substring = "Python"
if substring in my_string:
    print(f"The substring '{substring}' exists in the string.")
else:
    print(f"The substring '{substring}' does not exist in the string.")

# Split the string into a list of words
words_list = my_string.split()
print(f"List of words: {words_list}")


# In[8]:


# question 8

# Create a list of numbers
numbers = [25, 68, 12, 89, 45, 37, 77, 54]

# Find the largest number in the list
largest = max(numbers)

# Find the smallest number in the list
smallest = min(numbers)

# Print the results
print(f"The largest number in the list is: {largest}")
print(f"The smallest number in the list is: {smallest}")


# In[ ]:




