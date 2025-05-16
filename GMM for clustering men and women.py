#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


# In[12]:


# Step 1: Generate synthetic height and weight data for men and women

np.random.seed(42)
# Men: Taller and heavier
men_height = np.random.normal(loc=178, scale=5, size=100)   # mean=178cm, std=5
men_weight = np.random.normal(loc=78, scale=8, size=100)    # mean=78kg, std=8

# Women: Shorter and lighter
women_height = np.random.normal(loc=165, scale=5, size=100)  # mean=165cm, std=5
women_weight = np.random.normal(loc=62, scale=6, size=100)   # mean=62kg, std=6

# Combine into one dataset
height = np.concatenate([men_height, women_height])
weight = np.concatenate([men_weight, women_weight])
X = np.column_stack((height, weight))  # shape (200, 2)


# In[13]:


# Step 2: Fit GMM (we assume 2 clusters: men and women)
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(X)


# In[14]:


# Step 3: Predict probabilities and hard labels
probs = gmm.predict_proba(X)  # soft assignments
labels = gmm.predict(X)       # hard labels (0 or 1)


# In[15]:


# Step 4: Plot
plt.figure(figsize=(10, 6))

# Show points with soft coloring based on probability
for i in range(len(X)):
    # Get probability of belonging to cluster 0 (men) and cluster 1 (women)
    p_men = probs[i][0]
    p_women = probs[i][1]
    
    # Interpolate color between blue (men) and pink (women)
    color = (p_women, 0.2, p_men)  # RGB: pinkish if p_women high, bluish if p_men high
    plt.scatter(X[i, 0], X[i, 1], color=color, s=40)
    # Interpolate color between blue (men) and pink (women)
    color = (p_women, 0.2, p_men)  # RGB: pinkish if p_women high, bluish if p_men high
    plt.scatter(X[i, 0], X[i, 1], color=color, s=40)

# Plot the GMM cluster centers
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], marker='x', color='black', s=100, label="Cluster Centers")

plt.title("GMM Clustering: Height vs. Weight (Men vs Women)")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.legend()
plt.grid(True)
plt.show()

