#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


# In[2]:


# 1. Load the Iris dataset
iris = load_iris()
X = iris.data[:, [0, 2]]  # We'll use sepal length and petal length for 2D visualization


# In[3]:


# 2. Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[4]:


# 3. Fit the Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X_scaled)


# In[5]:


# 4. Predict anomalies
y_pred = model.predict(X_scaled)


# In[6]:


# 5. Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[y_pred == 1][:, 0], X_scaled[y_pred == 1][:, 1], 
            c='blue', label='Normal')
plt.scatter(X_scaled[y_pred == -1][:, 0], X_scaled[y_pred == -1][:, 1], 
            c='red', label='Anomaly', marker='x')
plt.xlabel('Sepal Length (standardized)')
plt.ylabel('Petal Length (standardized)')
plt.title('Isolation Forest on Iris Dataset')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




