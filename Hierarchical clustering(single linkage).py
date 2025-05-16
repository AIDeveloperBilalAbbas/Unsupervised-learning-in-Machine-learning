#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering


# In[7]:


# 1. Sample 2D data
X = np.array([
    [1, 2],
    [2, 3],
    [3, 3],
    [8, 7],
    [8, 8],
    [25, 80]
])


# In[8]:


# 2. Linkage matrix using 'single' linkage
linked = linkage(X, method='single')


# In[9]:


# 3. Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linked,
           orientation='top',
           labels=np.arange(1, len(X)+1),
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram (Single Linkage)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.grid(True)
plt.show()


# In[14]:


# 4. Apply Agglomerative Clustering with single linkage
cluster = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='single')
y_pred = cluster.fit_predict(X)


# In[19]:


# 5. Plot clustered data
plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='rainbow')
plt.title('Agglomerative Clustering (Single Linkage - 3 Clusters)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()


# In[ ]:




