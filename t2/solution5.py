import nltk
from nltk.corpus import movie_reviews

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the movie_reviews dataset from NLTK
#    Each review is stored as a list of words in the corpus;
#    categories can be 'pos' or 'neg'.
documents = []
labels = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        # Join tokens to reconstruct the full review text
        review_text = " ".join(movie_reviews.words(fileid))
        documents.append(review_text)
        labels.append(category)

# Convert labels to numeric: 'pos' -> 1, 'neg' -> 0
y = [1 if label == 'pos' else 0 for label in labels]

# 2. Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(documents, y, test_size=0.2, random_state=42)

# 3. Convert text into numerical features (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# 4. Train a Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_tfidf, y_train)

# 5. Make predictions on the test set
y_pred = log_reg.predict(X_test_tfidf)

# 6. Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['neg', 'pos']))

# (Optional) Display a few random predictions vs actual labels
import random
sample_indices = random.sample(range(len(X_test)), 5)
for i in sample_indices:
    print("\nReview snippet:")
    print(X_test[i][:300] + '...')  # Display first 300 characters of the review
    print("Actual Label:", "pos" if y_test[i] == 1 else "neg")
    print("Predicted Label:", "pos" if y_pred[i] == 1 else "neg")
