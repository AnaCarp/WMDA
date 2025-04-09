import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 1. Generate synthetic customer data
#    For example:
#    - 'purchase_frequency': how many purchases per month
#    - 'average_spent': average amount spent per purchase
#    - 'loyalty_score': a simple 1–5 rating

np.random.seed(42)
num_customers = 50

df_customers = pd.DataFrame({
    'purchase_frequency': np.random.randint(1, 15, num_customers),
    'average_spent': np.random.randint(10, 500, num_customers),
    'loyalty_score': np.random.randint(1, 6, num_customers)
})

print("=== Raw Customer Data (first 5 rows) ===")
print(df_customers.head(), "\n")

# 2. Scale the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_customers)

# 3. K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df_customers['cluster'] = kmeans.fit_predict(df_scaled)

# 4. Add cluster labels to the DataFrame
print("\n=== Customer Segments (with cluster labels) ===")
print(df_customers.head(), "\n")

# 5. Inspect each segment
segment_summary = df_customers.groupby('cluster').agg({
    'purchase_frequency': ['mean', 'std'],
    'average_spent': ['mean', 'std'],
    'loyalty_score': ['mean', 'std']
})

print("\n=== Segment Summary ===")
print(segment_summary)

# 6. (Optional) Quick interpretation or marketing actions
# Example: Interpreting each cluster based on behavior
for i in range(3):
    segment = df_customers[df_customers['cluster'] == i]
    print(f"\nCluster {i}:")
    if i == 0:
        print("High spending, frequent buyers. Marketing: Loyalty programs, premium offers.")
    elif i == 1:
        print("Moderate spending, moderate frequency. Marketing: Email offers, discounts.")
    else:
        print("Low spending, infrequent buyers. Marketing: Retargeting, special deals for engagement.")

# Optional: Visualize clusters in 2D for better understanding (using first two features)
plt.figure(figsize=(8, 6))
plt.scatter(df_customers['purchase_frequency'], df_customers['average_spent'], c=df_customers['cluster'], cmap='viridis')
plt.title("Customer Segments")
plt.xlabel('Purchase Frequency')
plt.ylabel('Average Spent')
plt.show()
