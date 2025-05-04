# Assignment No: 1
# Name: Pushkar Mahajan
# Roll No: 33540
# Batch: AIML - A
# Title: Titanic Dataset Analysis and Survival Prediction
# Problem Statement: Analyze the Titanic dataset to explore factors affecting survival and build a logistic regression model to predict passenger survival.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = sns.load_dataset('titanic')

# Data Preprocessing
# Handle missing values
df['age'].fillna(df['age'].mean(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
# Drop irrelevant columns (if they exist in the dataset)
df.drop(columns=['deck', 'who', 'adult_male', 'embark_town', 'alive', 'class', 'alone'], errors='ignore', inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, columns=['sex', 'embarked'], drop_first=True)

# Define features and target
X = df.drop('survived', axis=1)
y = df['survived']

# Scale numerical features
scaler = StandardScaler()
X[['age', 'fare']] = scaler.fit_transform(X[['age', 'fare']])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Visualizations
# 1. Survival count by gender
plt.figure(figsize=(8, 6))
sns.countplot(x='survived', hue='sex_male', data=df)
plt.title('Survival Count by Gender\nPushkar Mahajan, Roll No: 33540')
plt.savefig('survival_by_gender.png')
plt.close()

# 2. Survival count by passenger class
plt.figure(figsize=(8, 6))
sns.countplot(x='survived', hue='pclass', data=df)
plt.title('Survival Count by Passenger Class\nPushkar Mahajan, Roll No: 33540')
plt.savefig('survival_by_pclass.png')
plt.close()

# Output: Screenshots of the above plots and accuracy will be saved as 'survival_by_gender.png' and 'survival_by_pclass.png'
print("Plots saved as 'survival_by_gender.png' and 'survival_by_pclass.png'")