#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.spatial.distance import cdist


# In[2]:


# 1. Sample data
data = {
    'AgeGroup': ['Young', 'Young', 'Middle', 'Middle', 'Older', 'Older',
                 'Young', 'Middle', 'Older', 'Young'],
    'Income': [30000, 35000, 50000, 55000, 40000, 42000, 31000, 52000, 41000, 32000],
    'SpendingScore': [70, 65, 50, 55, 30, 25, 80, 45, 20, 75]
}


# In[3]:


df = pd.DataFrame(data)


# In[4]:


df


# In[5]:


# 2. Encode categorical variable
label_encoder = LabelEncoder()
df['AgeGroupEncoded'] = label_encoder.fit_transform(df['AgeGroup'])


# In[6]:


df


# In[7]:


# 3. Features for clustering
features = df[['AgeGroupEncoded', 'Income', 'SpendingScore']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# In[8]:


# 4. Average linkage hierarchical clustering
linked = linkage(features_scaled, method='average')


# In[9]:


# 5. Assign clusters (e.g., 3 clusters)
df['Cluster'] = fcluster(linked, 3, criterion='maxclust')


# In[10]:


# 6. Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linked,
           labels=df.index.tolist(),
           leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram (Average Linkage)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.axhline(y=7, color='r', linestyle='--')  # threshold line
plt.show()


# In[11]:


# 7. Visualize the clusters
sns.scatterplot(data=df, x='Income', y='SpendingScore', hue='Cluster', palette='Set1', s=100)
plt.title('Customer Segments (Average Linkage)')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.grid(True)
plt.show()


# In[ ]:




