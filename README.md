# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Path to your dataset
file_path = r'C:\Users\Pavan\Downloads\Documents\Blood_samples_dataset_balanced_2(f).csv'  # Use raw string to avoid escape characters

# Load the dataset (replace 'health_data.csv' with your actual dataset file name)
df = pd.read_csv(r'C:\Users\Pavan\Downloads\Documents\Blood_samples_dataset_balanced_2(f).csv)

# Display first few rows of the dataset
print(df.head())

# Assuming the dataset has the following columns:
# 'Age', 'Gender', 'Cholesterol', 'Blood Sugar', 'BMI', 'Family History', 'Heart Attack', 'Diabetes'

# Features (X) - excluding the target variables
X = df.drop(columns=['Heart Attack', 'Diabetes'])

# Target labels (Y) - 'Heart Attack' and 'Diabetes'
Y = df[['Heart Attack', 'Diabetes']]

# Encoding categorical variables (e.g., Gender, Family History if they are categorical)
label_encoder = LabelEncoder()
for column in X.select_dtypes(include=['object']).columns:
    X[column] = label_encoder.fit_transform(X[column])

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier (You can try other classifiers as well)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, Y_train)

# Predict on the test set
Y_pred = model.predict(X_test)

# Evaluate the model's performance for each disease
for i, column in enumerate(Y.columns):
    print(f"Performance for {column}:")
    print(f"Accuracy: {accuracy_score(Y_test[column], Y_pred[:, i]):.4f}")
    print(classification_report(Y_test[column], Y_pred[:, i]))
    print("\n")

# Visualizing feature importances
feature_importances = model.feature_importances_
features = X.columns

# Create a DataFrame for easier visualization
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plotting feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importance for Heart Attack and Diabetes Prediction")
plt.show()
