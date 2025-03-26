## Exercise 3 (10 minutes): Regression Trees
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. Create a synthetic dataset with multiple features
np.random.seed(42)
num_samples = 30
X = np.random.rand(num_samples, 3) * 10  # e.g., three numeric features

# Let's define a "true" relationship for the target:
# Target = 2*Feature1 + 0.5*Feature2^2 - 3*Feature3 + noise
true_y = 2 * X[:, 0] + 0.5 * (X[:, 1]**2) - 3 * X[:, 2]
noise = np.random.normal(0, 5, size=num_samples)  # Add some noise
y = true_y + noise

# Convert to a pandas DataFrame for familiarity
df = pd.DataFrame(X, columns=["Feature1", "Feature2", "Feature3"])
df["Target"] = y

# 2. Separate features and target
X_features = df[["Feature1", "Feature2", "Feature3"]]
y_target = df["Target"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3, random_state=42)

# 4. Create and train the Decision Tree Regressor
#    You can tune hyperparameters like max_depth to control overfitting
tree_model = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

# 5. Evaluate on the test set
y_pred = tree_model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Optional: Inspect feature importances
print("Feature Importances:", tree_model.feature_importances_)

# Optional: You could visualize the tree with:
# from sklearn.tree import export_graphviz