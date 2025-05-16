#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[12]:


# 1. Create dummy data
# We'll create a 2D dataset where the second feature is highly correlated with the first
np.random.seed(0)
x1 = np.random.normal(5, 1, 100)
x2 = x1 * 0.5 + np.random.normal(0, 0.2, 100)  # x2 is roughly 0.5 * x1 + some noise
X = np.column_stack((x1, x2))


# In[18]:


# 2. Standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)


# In[16]:


# 3. Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_std)


# In[9]:


# 4. Plot the original data
plt.figure(figsize=(14, 6))
# Original data
plt.subplot(1, 2, 1)
plt.scatter(X_std[:, 0], X_std[:, 1], alpha=0.6)
plt.title("Original Standardized Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
# Transformed data
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, color='green')
plt.title("Data After PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)

plt.tight_layout()
plt.show()


# In[ ]:




