import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
import json

# Debug Log: Start script
print("Script started...")

# Load features and activity labels
print("Loading features and activity labels...")
features_path = r"human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\features.txt"
activity_labels_path = r"human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\activity_labels.txt"
features = pd.read_csv(features_path, delim_whitespace=True, header=None, names=['index', 'feature'])
activity_labels = pd.read_csv(activity_labels_path, delim_whitespace=True, header=None, names=['label', 'activity'])

# Debug Log: Show loaded data
print("Features loaded:")
print(features.head())
print("Activity labels loaded:")
print(activity_labels.head())

# Load training and test datasets
print("Loading training and test datasets...")
X_train_path = r"human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train\X_train.txt"
y_train_path = r"human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\train\y_train.txt"
X_test_path = r"human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\X_test.txt"
y_test_path = r"human+activity+recognition+using+smartphones\UCI HAR Dataset\UCI HAR Dataset\test\y_test.txt"

X_train = pd.read_csv(X_train_path, delim_whitespace=True, header=None)
y_train = pd.read_csv(y_train_path, delim_whitespace=True, header=None, names=['label'])
X_test = pd.read_csv(X_test_path, delim_whitespace=True, header=None)
y_test = pd.read_csv(y_test_path, delim_whitespace=True, header=None, names=['label'])

# Debug Log: Show dataset sizes
print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Test data shape: {X_test.shape}, {y_test.shape}")

# Merge with activity labels for better readability
print("Merging activity labels with training and test labels...")
y_train = y_train.merge(activity_labels, on='label', how='left')
y_test = y_test.merge(activity_labels, on='label', how='left')

# Combine training and test sets for processing
print("Combining training and test datasets...")
X = pd.concat([X_train, X_test])
y = pd.concat([y_train, y_test])

# Train-test split (80-20)
print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X, y['activity'], test_size=0.2, random_state=42)

# Start MLflow tracking
print("Starting MLflow tracking for Iteration 1...")
mlflow.start_run(run_name="Iteration 1")

# Initialize and train a Random Forest Classifier
print("Training first Random Forest model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Log model and parameters to MLflow
print("Logging metrics and model to MLflow for Iteration 1...")
mlflow.log_param("n_estimators", 100)
mlflow.log_param("random_state", 42)
mlflow.sklearn.log_model(clf, "random_forest_model_iteration1")

# Predictions
print("Making predictions on validation data...")
y_pred = clf.predict(X_val)

# Evaluate the model
print("Evaluation Results for Iteration 1:")
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("Classification Report:\n", classification_report(y_val, y_pred))
mlflow.log_metric("val_accuracy", clf.score(X_val, y_val))

# Save the first model
joblib.dump(clf, "har_model.pkl")
print("First model saved as 'har_model.pkl'.")

# End MLflow run
mlflow.end_run()
print("Iteration 1 completed and logged to MLflow.")

# Start MLflow tracking for the second iteration
print("Starting MLflow tracking for Iteration 2...")
mlflow.start_run(run_name="Iteration 2")

# Train the second model
print("Training second Random Forest model with updated hyperparameters...")
clf_v2 = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
clf_v2.fit(X_train, y_train)

# Log parameters and save the second model
print("Logging metrics and model to MLflow for Iteration 2...")
mlflow.log_param("n_estimators", 200)
mlflow.log_param("max_depth", 10)
mlflow.sklearn.log_model(clf_v2, "random_forest_model_iteration2")

# Predictions for the second iteration
print("Making predictions on validation data for Iteration 2...")
y_pred_v2 = clf_v2.predict(X_val)

# Evaluate the second model
print("Evaluation Results for Iteration 2:")
print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred_v2))
print("Classification Report:\n", classification_report(y_val, y_pred_v2))
mlflow.log_metric("val_accuracy", clf_v2.score(X_val, y_val))

# Save the second model
joblib.dump(clf_v2, "random_forest_model_v2.pkl")
print("Second model saved as 'random_forest_model_v2.pkl'.")

# End MLflow run
mlflow.end_run()
print("Iteration 2 completed and logged to MLflow.")

# Save feature importance
print("Saving feature importance to 'feature_importance.csv'...")
feature_importance = pd.DataFrame({
    'Feature': features['feature'],
    'Importance': clf.feature_importances_
}).sort_values(by='Importance', ascending=False)
feature_importance.to_csv("feature_importance.csv", index=False)
print("Feature importance saved.")

print("Script executed successfully!")
