import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Optional: Convert to a DataFrame for exploration
# df = pd.DataFrame(X, columns=wine.feature_names)
# df['target'] = y
# print(df.head())

# 2. Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train a Decision Tree Classifier with max_depth=3 to control overfitting
tree_classifier = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_classifier.fit(X_train, y_train)

# 4. Check accuracy on the test set
y_pred = tree_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# (Optional) Visualize the tree structure
plt.figure(figsize=(12, 8))
plot_tree(tree_classifier, feature_names=wine.feature_names, class_names=wine.target_names, filled=True)
plt.title("Decision Tree Structure")
plt.show()

# (Optional) Display feature importances
importances = tree_classifier.feature_importances_
print("Feature importances:", importances)
