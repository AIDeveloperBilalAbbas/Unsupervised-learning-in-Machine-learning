#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler


# In[11]:


# 1. Generate synthetic customer data
np.random.seed(42)

# Simulate data for 3 age groups
n_samples = 50
young = pd.DataFrame({
    'AgeGroup': ['Young'] * n_samples,
    'Income': np.random.normal(30000, 5000, n_samples),
    'SpendingScore': np.random.normal(70, 10, n_samples)
})
middle = pd.DataFrame({
    'AgeGroup': ['Middle'] * n_samples,
    'Income': np.random.normal(50000, 7000, n_samples),
    'SpendingScore': np.random.normal(50, 12, n_samples)
})

older = pd.DataFrame({
    'AgeGroup': ['Older'] * n_samples,
    'Income': np.random.normal(40000, 6000, n_samples),
    'SpendingScore': np.random.normal(30, 8, n_samples)
})

# Combine data
df = pd.concat([young, middle, older], ignore_index=True)
df.head()


# In[3]:


# 2. Encode categorical data
label_encoder = LabelEncoder()
df['AgeGroupEncoded'] = label_encoder.fit_transform(df['AgeGroup'])


# In[4]:


# 3. Features for clustering
features = df[['AgeGroupEncoded', 'Income', 'SpendingScore']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# In[5]:


# 4. Hierarchical Clustering with Complete Linkage
linked = linkage(features_scaled, method='complete')


# In[10]:


# 5. Assign clusters (let's say we want 3 clusters)
df['Cluster'] = fcluster(linked, 3, criterion='maxclust')
df.head()


# In[7]:


# 6. Plot dendrogram
plt.figure(figsize=(12, 6))
dendrogram(linked, labels=df['AgeGroup'].values, leaf_rotation=90, color_threshold=10)
plt.title('Hierarchical Clustering Dendrogram (Complete Linkage)')
plt.xlabel('Sample Index or Age Group')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()


# In[8]:


# 7. Plot clusters (2D scatter using SpendingScore vs Income)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Income', y='SpendingScore', hue='Cluster', palette='Set1', style='AgeGroup')
plt.title('Customer Clusters Based on Behavior')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




