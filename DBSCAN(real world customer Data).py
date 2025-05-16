#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


# In[33]:


# Step 1: Load real-world dataset
df = pd.read_csv("Mall_Customers.csv")  # Make sure it's in your working directory
df.head()


# In[11]:


# Step 2: Select features (we'll use Annual Income and Spending Score)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values


# In[26]:


# Step 3: Standardize the data (important for DBSCAN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[19]:


# Step 4: Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=6)
labels = dbscan.fit_predict(X_scaled)


# In[35]:


# Step 5: Add cluster labels to the original dataframe
df['Cluster'] = labels
df.head()


# In[36]:


# Step 6: Define human-readable names for each cluster
cluster_labels = {
    0: "Impulsive Shoppers (Low Income, High Spend)",
    1: "Careful Spenders (High Income, Low Spend)",
    2: "Middle Class (Avg Income & Spend)",
    3: "Budget Shoppers (Low Income, Low Spend)",
    4: "Premium Customers (High Income, High Spend)",
    -1: "Noise"
}
df['Cluster Label'] = df['Cluster'].map(cluster_labels)
df.head()


# In[31]:


# Step 7: Show stats for each group
print("\nCluster Behavior:\n")
print(df.groupby('Cluster Label')[['Annual Income (k$)', 'Spending Score (1-100)']].mean())


# In[32]:


# Step 8: Plot the clusters with custom labels
plt.figure(figsize=(10, 7))
unique_labels = set(labels)

for label in unique_labels:
    mask = df['Cluster'] == label
    label_name = cluster_labels.get(label, f"Cluster {label}")
    color = 'k' if label == -1 else plt.cm.tab10(label)
    marker = 'x' if label == -1 else 'o'
    
    plt.scatter(
        df.loc[mask, 'Annual Income (k$)'],
        df.loc[mask, 'Spending Score (1-100)'],
        c=[color],
        label=label_name,
        marker=marker,
   s=100,
        edgecolors='k'
    )

plt.title("Customer Segmentation using DBSCAN")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




