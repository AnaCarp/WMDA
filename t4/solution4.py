## Exercise 4 (10 minutes): Agglomerative Clustering & Dendrogram

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# -- SIMULATION OF PREPROCESSED DATA (Replace this block with your actual df_scaled) --
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
# ------------------------------------------------------------------------------------

# 2. Perform Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg_clustering.fit_predict(df_scaled)

# 3. Add the cluster labels to the DataFrame
df_scaled['Cluster'] = labels

# 4. Print a quick summary of how many points were assigned to each cluster
print(f"Cluster distribution:\n{df_scaled['Cluster'].value_counts()}")

# 5. Create a linkage matrix for plotting a dendrogram
linkage_matrix = linkage(df_scaled.iloc[:, :-1], method='ward')

# 6. Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix)
plt.title('Agglomerative Clustering Dendrogram')
plt.xlabel('Data points')
plt.ylabel('Euclidean distance')
plt.show()
