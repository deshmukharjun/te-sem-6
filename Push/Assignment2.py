# Assignment No: 2
# Name: Pushkar Mahajan 
# Roll No: 33540
# Batch: AIML - A
# Title: Twitter Sentiment Analysis with K-Nearest Neighbors
# Problem Statement: Analyze a Twitter dataset to classify tweets as positive or negative using the K-Nearest Neighbors algorithm and evaluate the model performance.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download NLTK resources (run once)
nltk.download('twitter_samples')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load Twitter dataset (using NLTK's twitter_samples)
from nltk.corpus import twitter_samples
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

# Create a DataFrame
tweets = positive_tweets + negative_tweets
labels = [1] * len(positive_tweets) + [0] * len(negative_tweets)
df = pd.DataFrame({'tweet': tweets, 'sentiment': labels})

# Data Preprocessing
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    # Tokenize and convert to lowercase
    tokens = word_tokenize(text.lower())
    # Remove punctuation and stopwords
    tokens = [t for t in tokens if t not in string.punctuation and t not in stop_words]
    return ' '.join(tokens)

df['cleaned_tweet'] = df['tweet'].apply(preprocess_text)

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_tweet']).toarray()
y = df['sentiment']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN model (k=5, Euclidean distance)
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics with name and roll number
print("Assignment 2: Twitter Sentiment Analysis")
print("Name: Pushkar Mahajan , Roll No: 33540")
print(f"Model Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix\nPushkar Mahajan , Roll No: 33540')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# Output: Screenshots of the console output (metrics) and confusion_matrix.png
print("Confusion matrix saved as 'confusion_matrix.png'")