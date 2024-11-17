#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np

# 1D Array using np.array
arr1d = np.array([1, 2, 3, 4, 5])
print("1D Array:", arr1d)

# 2D Array using np.arange
arr2d = np.arange(12).reshape(3, 4)
print("2D Array:", arr2d)

# 3D Array using np.linspace
arr3d = np.linspace(0, 1, 27).reshape(3, 3, 3)
print("3D Array:", arr3d)

# Indexing and Slicing
print("Element at index 2:", arr1d[2])
print("Slice of 2D Array:", arr2d[1:3, 2:])

# Reshaping
arr2d_reshaped = arr2d.reshape(6, 2)
print("Reshaped 2D Array:", arr2d_reshaped)

# Concatenation
arr_concat = np.concatenate((arr1d, arr1d))
print("Concatenated Array:", arr_concat)

# Array Attributes
print("Shape:", arr1d.shape)
print("Size:", arr1d.size)
print("Data Type:", arr1d.dtype)
print("Number of Dimensions:", arr1d.ndim)

# Methods
print("Reshaped 2D Array:", arr2d.reshape(2, 6))
print("Resized 2D Array:", arr2d.resize(2, 6))
print("Flattened Array:", arr2d.flatten())


# In[7]:


# Import necessary libraries
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Load the iris dataset
iris = datasets.load_iris()
data = iris.data

# Convert to NumPy array
data_np = np.array(data)

# 1. Data Cleaning: Handle missing values (if any)
# For this example, we introduce some missing values to demonstrate imputation
data_np[0, 0] = np.nan  # Inject a missing value for demonstration

# Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
data_clean = imputer.fit_transform(data_np)

# 2. Normalization
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data_clean)

# 2. Standardization
scaler_std = StandardScaler()
data_standardized = scaler_std.fit_transform(data_clean)

# 3. Calculate statistical measures
mean = np.mean(data_clean, axis=0)
median = np.median(data_clean, axis=0)
std_dev = np.std(data_clean, axis=0)
variance = np.var(data_clean, axis=0)

# Print statistical measures
print("Mean:\n", mean)
print("Median:\n", median)
print("Standard Deviation:\n", std_dev)
print("Variance:\n", variance)


# In[ ]:




