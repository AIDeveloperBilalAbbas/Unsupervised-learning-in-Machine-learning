#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[4]:


# Step 1: Create dummy customer data
np.random.seed(42)  # for reproducibility
data = {
    'Age': np.random.randint(18, 60, 100),
    'Annual Income (k$)': np.random.randint(20, 120, 100),
    'Spending Score (1-100)': np.random.randint(1, 100, 100)
}

df = pd.DataFrame(data)


# In[6]:


df.head()


# In[7]:


# Step 2: Preprocess data (scaling)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)


# In[8]:


# Step 3: Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)


# In[19]:


# Step 4: Define descriptive labels for each cluster
cluster_descriptions = {
    0: 'Budget-Conscious Youth',
    1: 'High Spenders',
    2: 'Average Shoppers',
    3: 'Mature Low Spenders'
}
df['Segment'] = df['Cluster'].map(cluster_descriptions)
df.head()


# In[12]:


# Step 5: Predict a new customer
new_customer = pd.DataFrame({
    'Age': [33],
    'Annual Income (k$)': [70],
    'Spending Score (1-100)': [85]
})

scaled_new = scaler.transform(new_customer)
new_cluster = kmeans.predict(scaled_new)[0]
new_segment = cluster_descriptions[new_cluster]

print(f"🧑 New customer belongs to segment: {new_segment} (Cluster {new_cluster})")


# In[13]:


# Step 6: Add new customer to df for plotting
new_customer['Cluster'] = new_cluster
new_customer['Segment'] = new_segment
df_full = pd.concat([df, new_customer], ignore_index=True)


# In[17]:


# 🎯 2D Visualization (Annual Income vs Spending Score) with Descriptive Labels
plt.figure(figsize=(10, 6))

# Plot original customers by segment
for i in range(4):
    cluster_points = df[df['Cluster'] == i]
    plt.scatter(cluster_points['Annual Income (k$)'],
                cluster_points['Spending Score (1-100)'],
                color=colors[i],
                label=cluster_descriptions[i],
                s=80)

# Plot new customer (black X)
plt.scatter(new_customer['Annual Income (k$)'],
            new_customer['Spending Score (1-100)'],
            color='black', marker='X', s=200, label=f'New Customer ({new_segment})')

# Plot cluster centroids
for i, center in enumerate(centroids):
    plt.scatter(center[1], center[2],
                marker='D', color=colors[i], s=100, edgecolor='black')

# Add descriptive text near centroids
for i, center in enumerate(centroids):
    plt.text(center[1] + 1, center[2], cluster_descriptions[i],
             fontsize=9, fontweight='bold', color=colors[i])

# Label axes and add legend
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('💰 2D Customer Segmentation with Descriptive Labels')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[14]:


# Step 7: 3D Visualization
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['red', 'green', 'blue', 'orange']
for i in range(4):
    cluster_points = df[df['Cluster'] == i]
    ax.scatter(cluster_points['Age'],
               cluster_points['Annual Income (k$)'],
               cluster_points['Spending Score (1-100)'],
               c=colors[i], label=cluster_descriptions[i], s=60)

# Plot the new customer
ax.scatter(new_customer['Age'], new_customer['Annual Income (k$)'],
           new_customer['Spending Score (1-100)'],
           c='black', marker='X', s=150, label='New Customer')

# Add centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
for i, center in enumerate(centroids):
    ax.scatter(center[0], center[1], center[2],
               marker='D', c=colors[i], s=100, edgecolor='black')

# Labels
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')
plt.title('🧍‍♂️ Customer Segmentation in 3D (K-Means)')
ax.legend()
plt.show()


# In[ ]:




