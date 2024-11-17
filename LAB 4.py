#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df_csv=pd.read_csv('Position_Salaries.csv')
print("CSV file opened successfully")

df_csv.to_excel('Position_Salaries.xlsx')
print("Excel file opened successfully")

df_csv.to_json('Position_Salaries.json')
print("Json file opened successfully")


# In[2]:


print(df_csv.shape)
print(df_csv.columns)
print(df_csv.dtypes)
print(df_csv.head())
print(df_csv.tail())
print(df_csv.info())
print(df_csv.describe())


# In[5]:


# Single column
column_data = df_csv['Salary']
print(column_data)
# Multiple columns
multiple_columns_data = df_csv[['Level', 'Salary']]
print(multiple_columns_data)


# In[8]:


row_data = df_csv.iloc[0]  
slice_data = df_csv[6:10]  

# Select rows by label
row_data_by_label = df_csv.loc[0] 
print(row_data_by_label)

# Select specific rows and columns
subset = df_csv.loc[0:4, ['Level', 'Salary']]
print(subset)

# Select rows and columns by position
subset = df_csv.iloc[0:5, [0, 1]]
print(subset)


# In[9]:


df = pd.DataFrame({
    'A': [1, 2, None, 4],
    'B': [None, 'b', 'c', 'd'],
    'C': [1.0, None, 3.0, 4.0]
})

missing_values = df.isnull()
print(missing_values)

missing_counts = df.isnull().sum()
print(missing_counts)


# In[11]:


df_filled = df.fillna({'A': 0, 'B': 'unknown', 'C': df['C'].mean()})
print(df_filled)


# In[14]:


dropped = df.dropna(axis=1)
print(dropped)


# In[15]:


df_interpolated = df.interpolate()
print(df_interpolated)


# In[17]:


# Min Max Scaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[['A', 'C']] = scaler.fit_transform(df[['A', 'C']])
print(df)


# In[18]:


# Z-Score Standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['A', 'C']] = scaler.fit_transform(df[['A', 'C']])
print(df)


# In[20]:


df = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'C']
})
# Create dummy variables
df_dummies = pd.get_dummies(df, columns=['Category'])
df_dummies = pd.get_dummies(df, columns=['Category'], drop_first=True)
print(df_dummies)


# In[21]:


df = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'B', 'C', 'C'],
    'Value': [10, 20, 30, 40, 50, 60]
})
# Group by 'Category' and calculate summary statistics
grouped = df.groupby('Category').agg({
    'Value': ['mean', 'median', 'count', 'sum']
})
print(grouped)


# In[22]:


df = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'B', 'C', 'C'],
    'Subcategory': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
    'Value': [10, 20, 30, 40, 50, 60]
})

# Create a pivot table
pivot_table = pd.pivot_table(df, values='Value', index='Category', columns='Subcategory', aggfunc='mean')

print(pivot_table)


# In[24]:


df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

# Concatenate along rows (axis=0)
concat_rows = pd.concat([df1, df2], axis=0)
# Concatenate along columns (axis=1)
concat_columns = pd.concat([df1, df2], axis=1)
print(concat_rows)
print(concat_columns)


# In[25]:


df1 = pd.DataFrame({'Key': ['A', 'B', 'C'], 'Value1': [1, 2, 3]})
df2 = pd.DataFrame({'Key': ['A', 'B', 'D'], 'Value2': [4, 5, 6]})

# Merge on 'Key'
merged = pd.merge(df1, df2, on='Key')
print(merged)


# In[26]:


df1 = pd.DataFrame({'Key': ['A', 'B', 'C'], 'Value1': [1, 2, 3]})
df2 = pd.DataFrame({'Key': ['A', 'B', 'D'], 'Value2': [4, 5, 6]})

# Merge on 'Key'
merged = pd.merge(df1, df2, on='Key')
print(merged)


# In[27]:


df1 = pd.DataFrame({'Value1': [1, 2, 3]}, index=['A', 'B', 'C'])
df2 = pd.DataFrame({'Value2': [4, 5]}, index=['A', 'B'])

# Join DataFrames
joined = df1.join(df2)
print(joined)


# In[33]:


df1 = pd.DataFrame({'Key': ['A', 'B', 'C'], 'Value1': [1, 2, 3]})
df2 = pd.DataFrame({'Key': ['A', 'B', 'D'], 'Value2': [4, 5, 6]})

# Inner Join
inner_join = pd.merge(df1, df2, on='Key', how='inner')
print(inner_join)

# Outer Join
outer_join = pd.merge(df1, df2, on='Key', how='outer')
print(outer_join)

#Left Join
left_join = pd.merge(df1, df2, on='Key', how='left')
print(left_join)

#Right_Join
right_join = pd.merge(df1, df2, on='Key', how='right')
print(right_join)


# In[ ]:




