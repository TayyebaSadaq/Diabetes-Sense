import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from lime.lime_tabular import LimeTabularExplainer
import warnings
import os

# Suppress all warnings
warnings.filterwarnings('ignore')

### IMPORTING PREPROCESSED DATA
base_path = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
data_path = os.path.join(base_path, "../data/balanced_pima.csv")  # Adjust path relative to script
data = pd.read_csv(data_path)

### Splitting the data into training and testing sets (80% training, 20% testing)
X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]  # Features
y = data['Outcome']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

### Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter grids
param_grid_lr = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga'],
    'class_weight': ['balanced', None]
}

param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced', None]
}

param_grid_gmb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Define Stratified K-Fold Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression with Cross-Validation
grid_search_lr = GridSearchCV(LogisticRegression(max_iter=500), param_grid_lr, cv=cv, scoring='accuracy')
grid_search_lr.fit(X_train_scaled, y_train)
best_lr = grid_search_lr.best_estimator_
cv_scores_lr = cross_val_score(best_lr, X_train_scaled, y_train, cv=cv, scoring='accuracy')
print(f"Logistic Regression Cross-Validation Accuracy: {cv_scores_lr.mean():.2f} ± {cv_scores_lr.std():.2f}")
y_pred_lr = best_lr.predict(X_test_scaled)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
classification_rep_lr = classification_report(y_test, y_pred_lr)
print(f"Logistic Regression Model (Best Parameters: {grid_search_lr.best_params_})")
print(f"Accuracy: {accuracy_lr:.2f}")
print("\nClassification Report:\n", classification_rep_lr)

# Random Forest with Cross-Validation
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=cv, scoring='accuracy')
grid_search_rf.fit(X_train_scaled, y_train)  # Use scaled data for consistency
best_rf = grid_search_rf.best_estimator_
cv_scores_rf = cross_val_score(best_rf, X_train_scaled, y_train, cv=cv, scoring='accuracy')
print(f"Random Forest Cross-Validation Accuracy: {cv_scores_rf.mean():.2f} ± {cv_scores_rf.std():.2f}")
y_pred_rf = best_rf.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_rep_rf = classification_report(y_test, y_pred_rf)
print(f"Random Forest Model (Best Parameters: {grid_search_rf.best_params_})")
print(f"Accuracy: {accuracy_rf:.2f}")
print("\nClassification Report:\n", classification_rep_rf)

# Gradient Boosting with Early Stopping
grid_search_gmb = GridSearchCV(
    GradientBoostingClassifier(n_iter_no_change=10, validation_fraction=0.1),  # Early stopping
    param_grid_gmb, cv=cv, scoring='accuracy'
)
grid_search_gmb.fit(X_train_scaled, y_train)
best_gmb = grid_search_gmb.best_estimator_
cv_scores_gmb = cross_val_score(best_gmb, X_train_scaled, y_train, cv=cv, scoring='accuracy')
print(f"Gradient Boosting Cross-Validation Accuracy: {cv_scores_gmb.mean():.2f} ± {cv_scores_gmb.std():.2f}")
y_pred_gmb = best_gmb.predict(X_test_scaled)
accuracy_gmb = accuracy_score(y_test, y_pred_gmb)
classification_rep_gmb = classification_report(y_test, y_pred_gmb)
print(f"Gradient Boosting Model (Best Parameters: {grid_search_gmb.best_params_})")
print(f"Accuracy: {accuracy_gmb:.2f}")
print("\nClassification Report:\n", classification_rep_gmb)

# Save the scaler
scaler_filename = os.path.join(base_path, "../models/scaler.pkl")
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved successfully at {scaler_filename}")

# Save the test data for evaluation
test_data = pd.concat([X_test, y_test], axis=1)
test_data_path = os.path.join(base_path, "../data/test_data.csv")
test_data.to_csv(test_data_path, index=False)
print(f"Test data saved successfully at {test_data_path}")

# Save the best models and their accuracies
models = [best_lr, best_rf, best_gmb]
model_names = ["logistic_regression.pkl", "random_forest.pkl", "gradient_boosting.pkl"]
accuracies = [cv_scores_lr.mean(), cv_scores_rf.mean(), cv_scores_gmb.mean()]

model_folder = os.path.join(base_path, "../models")

# Save each model and its accuracy
for each, name, accuracy in zip(models, model_names, accuracies):
    model_path = os.path.join(model_folder, name)
    joblib.dump((each, accuracy), model_path)
    print(f"{name} saved successfully at {model_folder} with accuracy {accuracy:.2f}")