#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import pandas as pd
import statistics
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv('bodyPerformance.csv')
df.head()


# In[3]:


numeric_data = df.select_dtypes(exclude='object')
categorical_data = df.select_dtypes(include='object')


# In[11]:


# Mean
a=df.mean()
print(a)
df['body fat_%'].mean()


# In[6]:


#Geometric Mean
from scipy.stats import gmean
gmean(df['body fat_%'])


# In[7]:


#Harmonic Mean
statistics.harmonic_mean(df['body fat_%'])


# In[8]:


#Mode
df.mode()


# In[9]:


#Median
statistics.median(df['body fat_%'])


# In[14]:


#Variance
a=statistics.variance(df['body fat_%'])
print(a)
statistics.variance(df['height_cm'])


# In[15]:


#Standard Deviation
statistics.stdev(df['body fat_%'])


# In[21]:


from scipy.stats import skew

numeric_data = df.select_dtypes(include=['float64', 'int64'])

skewness = numeric_data.apply(skew)
print(skewness)


# In[24]:


#Interquartile Range (IQR)
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[27]:


# Range: Difference between max and min.
numeric_df=df.select_dtypes(include=['float64','int64'])

df_range=numeric_df.max()-numeric_df.min()
print("Range of numeric columns:")
print(df_range)


# In[30]:


# Mean Absolute Deviation (MAD)
numeric_df = df.select_dtypes(include=['float64', 'int64'])
mean = numeric_df.mean()
print("Mean values of numeric columns:")
print(mean)


# In[32]:


# Correlation Matrix
correlation_matrix = df.corr()
print(correlation_matrix)


# In[33]:


pearson_corr = df.corr(method='pearson')
spearman_corr = df.corr(method='spearman')


# In[34]:


#Boxplot
sns.boxplot(data=df)


# In[37]:


# Histogram
plt.figure(figsize=(10, 6))
for column in numeric_df.columns:
    plt.hist(numeric_df[column], bins=30, alpha=0.5, label=column)

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Combined Histogram of Numeric Columns')
plt.legend()
plt.show()


# In[39]:


# Density Plot
sns.kdeplot(df, shade=True)


# In[44]:


# Scatter Plot
plt.subplot(2, 2, 4)
sns.scatterplot(x='age', y='weight_kg', data=df)
plt.title("Age vs Weight Scatterplot")


# In[45]:


# BarGraph
plt.figure(figsize=(6,4))
sns.countplot(x='gender', data=df)  # Change 'gender' to a relevant categorical column
plt.title('Gender Distribution')
plt.show()
plt.tight_layout()
plt.show()


# In[ ]:




