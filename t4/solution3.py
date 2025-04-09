import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

# -- SIMULATION OF PREPROCESSED DATA (Replace this block with your actual df_scaled) --
iris = load_iris()
df_scaled = pd.DataFrame(iris.data, columns=iris.feature_names)
# ------------------------------------------------------------------------------------

# 2. Instantiate DBSCAN with chosen parameters
# eps defines the neighborhood radius, min_samples is the minimum number of points
# for a region to be considered dense.
dbscan = DBSCAN(eps=0.5, min_samples=5)

# 3. Fit the model to the data
dbscan.fit(df_scaled)

# 4. Extract cluster labels
labels_dbscan = dbscan.labels_

# 5. Identify outliers (DBSCAN labels outliers as -1)
outliers = np.sum(labels_dbscan == -1)
clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)

# 6. (Optional) Add the labels to the DataFrame
df_scaled['DBSCAN Cluster'] = labels_dbscan

# 7. Print the cluster label counts
print(f"Number of clusters detected: {clusters}")
print(f"Number of outliers: {outliers}")

# 8. Optional quick visualization (for 2D only)
#    Choose two features to plot, coloring by DBSCAN labels
plt.figure(figsize=(8, 6))
plt.scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1], c=labels_dbscan, cmap='viridis')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('DBSCAN Clustering')
plt.colorbar(label='Cluster label')
plt.show()

# Optional: Histogram of DBSCAN cluster frequencies (including outliers)
plt.figure(figsize=(8, 6))
plt.hist(labels_dbscan, bins=np.arange(-1, clusters + 1) - 0.5, edgecolor='black')
plt.xlabel('Cluster Label')
plt.ylabel('Frequency')
plt.title('Histogram of DBSCAN Cluster Frequencies')
plt.xticks(np.arange(-1, clusters + 1))
plt.show()
