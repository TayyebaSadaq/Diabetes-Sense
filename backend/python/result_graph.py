import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the models
model_folder = r"C:\Users\tayye\Desktop\Diabetes-Prediction-using-Machine-Learning-and-Explainable-AI-Techniques\diabetes-sense\app\models"
model_names = ["logistic_regression.pkl", "random_forest.pkl", "gradient_boosting.pkl"]
models = [joblib.load(model_folder + "\\" + name)[0] for name in model_names]

# Load the data
data = pd.read_pickle(r"C:\Users\tayye\Desktop\Diabetes-Prediction-using-Machine-Learning-and-Explainable-AI-Techniques\diabetes-sense\app\data\preprocessed_pima.pkl")
X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = data['Outcome']

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the test data to a CSV file for manual testing
X_test_with_labels = X_test.copy()
X_test_with_labels['Outcome'] = y_test
X_test_with_labels.to_csv(r"C:\Users\tayye\Desktop\Diabetes-Prediction-using-Machine-Learning-and-Explainable-AI-Techniques\diabetes-sense\app\data\test_data.csv", index=False)
print("Test data saved to test_data.csv")

# Calculate accuracies and confidence scores
accuracy_train = []
accuracy_test = []
confidence_scores = []

for model in models:
    if isinstance(model, RandomForestClassifier):
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        confidence = np.mean([max(proba) for proba in model.predict_proba(X_test)])
    else:
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        confidence = np.mean([max(proba) for proba in model.predict_proba(X_test_scaled)])
    
    accuracy_train.append(accuracy_score(y_train, y_train_pred))
    accuracy_test.append(accuracy_score(y_test, y_test_pred))
    confidence_scores.append(confidence)

# Sample LIME scores (replace with actual LIME scores if available)
lime_scores = [0.85, 0.80, 0.82]

# Print the calculated values for verification
print("Training Accuracies:", accuracy_train)
print("Test Accuracies:", accuracy_test)
print("Confidence Scores:", confidence_scores)
print("LIME Scores:", lime_scores)

# Plotting accuracy/confidence comparison
plt.figure(figsize=(18, 6))

# Training and Test Accuracy
plt.subplot(1, 3, 1)
x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, accuracy_train, width, label='Train Accuracy')
plt.bar(x + width/2, accuracy_test, width, label='Test Accuracy')

plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.xticks(x, ['Logistic Regression', 'Random Forest', 'Gradient Boosting'])
plt.legend()

# Confidence Scores Comparison
plt.subplot(1, 3, 2)
sns.barplot(x=['Logistic Regression', 'Random Forest', 'Gradient Boosting'], y=confidence_scores)

plt.xlabel('Models')
plt.ylabel('Confidence Score')
plt.title('Model Confidence Comparison')

# LIME Explainable AI Comparison
plt.subplot(1, 3, 3)
sns.barplot(x=['Logistic Regression', 'Random Forest', 'Gradient Boosting'], y=lime_scores)

plt.xlabel('Models')
plt.ylabel('LIME Score')
plt.title('LIME Explainable AI Comparison')

plt.tight_layout()
plt.show()