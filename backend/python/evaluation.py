import os
import pickle
import joblib  # Add joblib for compatibility
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# Define the folder where the models are stored
model_folder = os.getenv('MODEL_FOLDER', os.path.join(os.path.dirname(__file__), '..', 'models'))

# Load the machine learning models and their accuracies
models = {}
accuracies = {}
model_names = ["logistic_regression.pkl", "random_forest.pkl", "gradient_boosting.pkl"]

for name in model_names:
    model_path = os.path.join(model_folder, name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    # Load the model and its accuracy from the file
    model, accuracy = joblib.load(model_path)
    models[name.split('.')[0]] = model  # Use the model name without the file extension as the key
    accuracies[name.split('.')[0]] = accuracy

# Load test data
test_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'test_data.csv')
if not os.path.exists(test_data_path):
    raise FileNotFoundError(f"Test data file not found: {test_data_path}. Please ensure the file exists at the specified location.")

test_data = pd.read_csv(test_data_path)
X_test = test_data.drop("Outcome", axis=1).values  # Convert to NumPy array to avoid feature name mismatch
y_test = test_data["Outcome"].values  # Convert to NumPy array for consistency

# Load the scaler
scaler_path = os.path.join(model_folder, "scaler.pkl")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found: {scaler_path}. Please ensure the scaler exists at the specified location.")
scaler = joblib.load(scaler_path)

# Scale the test data
X_test = scaler.transform(X_test)

# Initialize LIME explainer
explainer = LimeTabularExplainer(
    training_data=X_test,
    feature_names=test_data.drop("Outcome", axis=1).columns.tolist(),
    class_names=["Non-Diabetic", "Diabetic"],
    mode="classification"
)

# Create evaluation folder if it doesn't exist
evaluation_folder = os.path.join(os.path.dirname(__file__), '..', 'evaluation')
os.makedirs(evaluation_folder, exist_ok=True)

# Evaluate each model
for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Debugging: Print predictions
    print(f"Predictions for {model_name}: {np.unique(y_pred, return_counts=True)}")

    # Print evaluation metrics
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for {model_name}")
    plt.legend(loc="lower right")
    roc_curve_path = os.path.join(evaluation_folder, f"{model_name}_roc_curve.png")
    plt.savefig(roc_curve_path)
    plt.close()

# Run LIME explanations on selected instances
selected_instances = pd.DataFrame(X_test, columns=test_data.drop("Outcome", axis=1).columns).sample(5, random_state=42)
for idx, instance in selected_instances.iterrows():
    explanation = explainer.explain_instance(
        data_row=instance.values,  # Use .values to avoid deprecated behavior
        predict_fn=models["logistic_regression"].predict_proba  # Change model as needed
    )
    lime_png_path = os.path.join(evaluation_folder, f"lime_explanation_{idx}.png")
    fig = explanation.as_pyplot_figure()

    # Customize the graph
    fig.set_size_inches(10, 8)  # Make the graph larger
    ax = fig.gca()
    ax.set_yticklabels([label.get_text().split(' ')[0] for label in ax.get_yticklabels()])  # Extract text and remove thresholds

    # Adjust layout to prevent feature names from being cut off
    fig.tight_layout(pad=3.0)
    plt.savefig(lime_png_path, bbox_inches="tight")  # Use bbox_inches to ensure everything fits
    plt.close()