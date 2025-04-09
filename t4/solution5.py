import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# 1. Load or assume you have a preprocessed dataset (df_scaled)
#    For demonstration, we'll again load & scale the Iris dataset
iris = load_iris()
X = iris.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Fit each clustering method
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
dbscan = DBSCAN(eps=0.5, min_samples=5).fit(X_scaled)
agg = AgglomerativeClustering(n_clusters=3, linkage='ward').fit(X_scaled)

# 3. Get the cluster labels from each method
labels_kmeans = kmeans.labels_
labels_dbscan = dbscan.labels_
labels_agg = agg.labels_

# 4. Compute silhouette scores (only if more than one cluster exists)
# DBSCAN might produce a single cluster or no clusters if parameters are not well-tuned,
# so we check to avoid an error in silhouette_score.
if len(set(labels_dbscan)) > 1:  # DBSCAN outliers may cause a single cluster
    silhouette_dbscan = silhouette_score(X_scaled, labels_dbscan)
else:
    silhouette_dbscan = -1  # If DBSCAN only creates one cluster, we set the score to -1

silhouette_kmeans = silhouette_score(X_scaled, labels_kmeans)
silhouette_agg = silhouette_score(X_scaled, labels_agg)

# 5. Print the scores
print(f"Silhouette score for K-Means: {silhouette_kmeans:.4f}")
print(f"Silhouette score for DBSCAN: {silhouette_dbscan:.4f}")
print(f"Silhouette score for Agglomerative Clustering: {silhouette_agg:.4f}")

# Optional: Plot silhouette scores for each method
import matplotlib.pyplot as plt
methods = ['K-Means', 'DBSCAN', 'Agglomerative Clustering']
scores = [silhouette_kmeans, silhouette_dbscan, silhouette_agg]

plt.bar(methods, scores, color=['blue', 'green', 'orange'])
plt.xlabel('Clustering Method')
plt.ylabel('Silhouette Score')
plt.title('Comparison of Silhouette Scores for Different Clustering Methods')
plt.show()
