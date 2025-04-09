import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

# -- SIMULATION OF PREPROCESSED DATA (Replace this block with your actual df_scaled) --
iris = load_iris()
df_scaled = pd.DataFrame(iris.data, columns=iris.feature_names)
# ------------------------------------------------------------------------------------

# 2. Instantiate K-Means with a chosen number of clusters, say 3
kmeans = KMeans(n_clusters=3, random_state=42)

# 3. Fit the model to the data
kmeans.fit(df_scaled)

# 4. Extract cluster labels
labels = kmeans.labels_

# 5. (Optional) Add the cluster labels to the DataFrame
df_scaled['Cluster'] = labels

# 6. Print or visualize the results
print("Cluster labels for each data point:")
print(df_scaled.head())  # Print the first few rows with cluster labels

# 7. Optional quick visualization (for 2D only)
#    If you'd like a scatter plot, choose two features to plot.
plt.figure(figsize=(8, 6))
plt.scatter(df_scaled.iloc[:, 0], df_scaled.iloc[:, 1], c=df_scaled['Cluster'], cmap='viridis')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.title('K-Means Clustering (3 clusters)')
plt.colorbar(label='Cluster label')
plt.show()

# Optional: Histogram of cluster frequencies
plt.figure(figsize=(8, 6))
plt.hist(labels, bins=np.arange(0, 4) - 0.5, edgecolor='black')
plt.xlabel('Cluster Label')
plt.ylabel('Frequency')
plt.title('Histogram of Cluster Frequencies')
plt.xticks([0, 1, 2])
plt.show()
