#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('Telecom_customer churn.csv') 
df.head()


# In[5]:


# Handling Missing Values

missing_values = df.isnull().sum() 
print(missing_values) 
df = df.dropna()


# In[3]:


# Get numeric columns 

numeric_cols = df.select_dtypes(include=['number']).columns 
print("Numeric columns:", numeric_cols)


# In[9]:


# Outlier Detection

outlier_dict = {}
for col in numeric_cols: # Calculate Q1 and Q3 
    Q1 = df[col].quantile(0.25) 
    Q3 = df[col].quantile(0.75) 
    IQR = Q3 - Q1 
    # Determine bounds for outliers 
    lower_bound = Q1 - 1.5 * IQR 
    upper_bound = Q3 + 1.5 * IQR 
    # Identify outliers 
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)] 
    if not outliers.empty: 
        outlier_dict[col] = outliers
# Display outliers for each column 
for col, outliers in outlier_dict.items(): 
    print(f"Outliers in column '{col}':") 
    print(outliers)


# In[11]:


# Remove outliers from the DataFrame 

for col in outlier_dict.keys(): 
    Q1 = df[col].quantile(0.25) 
    Q3 = df[col].quantile(0.75) 
    IQR = Q3 - Q1 
    lower_bound = Q1 - 1.5 * IQR 
    upper_bound = Q3 + 1.5 * IQR 
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    print(df)


# In[31]:


# Feature Selection

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
scaled_features = scaler.fit_transform(df[numeric_cols]) # Replace with your numeric columns 
df_scaled = pd.DataFrame(scaled_features, columns=numeric_cols)
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from scipy.stats import zscore
df.hist(bins=20, figsize=(15, 10)) 
plt.tight_layout() 
plt.show()


# In[32]:


# Exploratory data analysis (EDA) to identify potential predictors 

# Visualizing the distribution of the target variable (assuming 'Churn' is a column in the dataset)
if 'Churn' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Churn', palette='Set2')
    plt.title('Distribution of Churn')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.show()

# Correlation heatmap to identify potential predictors
plt.figure(figsize=(15, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap')
plt.show()

# Visualize relationships between some key features and the target variable
if 'Churn' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='Churn', y='rev_Mean')
    plt.title('Revenue Mean vs. Churn')
    plt.xlabel('Churn')
    plt.ylabel('Revenue Mean')
    plt.show()


# In[36]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X = df.drop('churn', axis=1)  # Features
y = df['churn']                # Target variable

# Convert categorical variables to numerical if necessary
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier (you can choose any classifier)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')


# In[33]:


# linear Regression
from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[34]:


from sklearn.linear_model import LinearRegression

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Predictions on the test set
y_pred = model.predict(X_test)


# In[35]:


from sklearn.metrics import mean_squared_error, r2_score

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R² Score: {r2}')


# In[ ]:




