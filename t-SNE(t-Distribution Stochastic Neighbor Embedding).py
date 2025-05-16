#!/usr/bin/env python
# coding: utf-8

# In[15]:


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[16]:


# Step 1: Load Iris dataset
iris = load_iris()
X = iris.data        # Shape: (150, 4)
y = iris.target      # Shape: (150,)


# In[17]:


# Step 2: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[18]:


# Step 3: PCA projection (4D → 2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# In[19]:


# Step 4: t-SNE projection (4D → 2D)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)


# In[20]:


# Step 5: Plot all 3
plt.figure(figsize=(18, 5))

# Subplot 1: Original 4D — we'll use 2 features to visualize
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 2], c=y, cmap='viridis', s=50)
plt.title('Original Data (2 Features)')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.grid(True)
# Subplot 2: PCA result
plt.subplot(1, 3, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50)
plt.title('PCA Projection (2D)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)

# Subplot 3: t-SNE result
plt.subplot(1, 3, 3)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', s=50)
plt.title('t-SNE Projection (2D)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True)

plt.suptitle("Iris Dataset: Original vs PCA vs t-SNE", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# Plot	What it Shows	Notes
# Left	Original data using 2 features (sepal length vs. petal length)	Just a slice of the real 4D data.
# Middle	PCA projection (2D)	Linear dimensionality reduction. Preserves variance.
# Right	t-SNE projection (2D)	Non-linear. Preserves local structure (better for visualizing clusters).

# Even though the original data has 4 features, both PCA and t-SNE give us useful 2D plots.
# 
# PCA gives a rough idea of separation.
# 
# t-SNE gives a much cleaner separation between flower types.

# In[ ]:




