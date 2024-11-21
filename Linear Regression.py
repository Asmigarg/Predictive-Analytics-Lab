#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


data=pd.read_csv('iris.csv') 
data.head()


# In[5]:


feature='sepal_length'
species=data['species'].unique()

data_to_plot=[data[data['species']==sp][feature] for sp in species]

plt.figure(figsize=(6,4))
plt.boxplot(data_to_plot,labels=species)
plt.title(f'Box plot of {feature} by species')
plt.xlabel('Species')
plt.ylabel(feature)

plt.show()


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        n = X.shape[0]
        for _ in range(epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = -(2/n) * np.dot(X.T, (y - y_pred))
            db = -(2/n) * np.sum(y - y_pred)
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

if __name__ == "__main__":
    # Load Iris dataset
    iris = pd.read_csv('iris.csv')
    print(iris.head())

    # Use sepal_length as feature (X) and petal_length as target (y)
    X = iris[['sepal_length']].values
    y = iris['petal_length'].values

    # Split dataset into training and testing sets
    train_size = int(0.8 * len(iris))
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train, epochs=1000, learning_rate=0.01)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Plot results
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, y_pred, color='red', label='Predicted', linewidth=2)
    plt.xlabel('Sepal Length')
    plt.ylabel('Petal Length')
    plt.title('Linear Regression on Iris Dataset')
    plt.legend()
    plt.show()

    # Print model parameters
    print("Weights:", model.weights)
    print("Bias:", model.bias)





