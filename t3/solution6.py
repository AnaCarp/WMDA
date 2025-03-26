## Exercise 6 (10 minutes): kNN for Regression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Create a synthetic dataset
np.random.seed(42)
num_samples = 30

# Let's generate two features (e.g., Feature1, Feature2) and a target
X = np.random.rand(num_samples, 2) * 10
# Define a "true" relationship for the target: y = 3*X1 + 2*X2 + noise
y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.normal(0, 5, size=num_samples)

# Convert to a DataFrame for clarity
df = pd.DataFrame(X, columns=["Feature1", "Feature2"])
df["Target"] = y

# 2. Separate features and target
X = df[["Feature1", "Feature2"]]
y = df["Target"]

# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Feature scaling (recommended for distance-based methods like kNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Create and train the kNN Regressor
# We'll start with n_neighbors=3 (can try different values)
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# 6. Evaluate on the test set
y_pred = knn.predict(X_test_scaled)
print("kNN Regression with k = 3")
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# 7. (Optional) Explore the effect of different k values
# You can loop over various values of k and compare performance.
print("\nExploring different values of k:")
for k in [1, 3, 5, 7]:
    knn_k = KNeighborsRegressor(n_neighbors=k)
    knn_k.fit(X_train_scaled, y_train)
    y_pred_k = knn_k.predict(X_test_scaled)
    print(f"\nk = {k}")
    print("R² Score:", r2_score(y_test, y_pred_k))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred_k))
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_k))


# Un k mai mic (de exemplu, k = 1) poate duce la supraadaptare, deoarece modelul este foarte influentat de punctele cele mai apropiate.
# Un k mai mare (de exemplu, k = 7) poate duce la subadaptare, deoarece predictiile devin prea netezite.
# Este important sa se scaleze caracteristicile in modelele bazate pe distanta pentru a evita ca diferentele de scara sa afecteze rezultatele.
# kNN realizeaza regresia prin medierea valorilor tinta ale vecinilor cei mai apropiati.
