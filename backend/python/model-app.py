# Import necessary libraries for the Flask app, machine learning, and data processing
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os
from sklearn.model_selection import GridSearchCV, cross_val_score

# Initialize the Flask app and enable CORS for cross-origin requests
app = Flask(__name__)
CORS(app)

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

# Load the scaler for preprocessing input data
scaler = joblib.load(os.path.join(model_folder, 'scaler.pkl'))

# Define the list of features expected in the input data
FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Load the training data for initializing the LIME explainer
data = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'balanced_pima.csv'))
X_train = data[FEATURES]

# Initialize the LIME explainer with the training data
explainer = LimeTabularExplainer(X_train.values, mode="classification", feature_names=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle POST requests to the /predict endpoint.
    This function takes input data, preprocesses it, and uses the selected models to make predictions.
    It also generates LIME explanations for the predictions.
    """
    try:
        # Parse the JSON input from the request
        data = request.get_json()
        input_data = [data.get(feature, 0) for feature in FEATURES]  # Default to 0 if a feature is missing
        input_df = pd.DataFrame([input_data], columns=FEATURES)
        
        # Scale the input data using the preloaded scaler
        input_df_scaled = scaler.transform(input_df)
        
        # Get the list of selected models from the input, or use all models by default
        selected_models = data.get('models', models.keys())
        results = {}
        
        # Iterate through the models and make predictions
        for model_name, model in models.items():
            if model_name not in selected_models:  # Skip models that are not selected
                continue

            # Handle scaling differences for Random Forest
            if model_name == "random_forest":
                prediction = model.predict(input_df)[0]
                confidence = max(model.predict_proba(input_df)[0])
                lime_input = input_df.values[0]
            else:
                prediction = model.predict(input_df_scaled)[0]
                confidence = max(model.predict_proba(input_df_scaled)[0])
                lime_input = input_df_scaled[0]

            # Convert the prediction to a human-readable result
            result = "Diabetic" if prediction == 1 else "Not Diabetic"

            # Generate a LIME explanation for the prediction
            exp = explainer.explain_instance(
                lime_input, model.predict_proba, num_features=8
            )
            lime_explanation = exp.as_list()

            # Define the fixed order of feature names for the graph
            fixed_feature_order = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            feature_importances_dict = {feature: 0 for feature in fixed_feature_order}  # Initialize all importances to 0

            # Update the feature importances based on the LIME explanation
            for feature, importance in lime_explanation:
                simplified_feature = feature.split(' ')[0]  # Simplify the feature name
                if simplified_feature in feature_importances_dict:
                    feature_importances_dict[simplified_feature] = importance

            # Prepare data for the graph
            simplified_feature_names = list(feature_importances_dict.keys())
            feature_importances = list(feature_importances_dict.values())

            # Create a bar chart visualization of the LIME explanation
            fig, ax = plt.subplots(figsize=(50, 30))  # Large graph size

            # Set bar colors: green for positive, red for negative importances
            bar_colors = ['green' if importance > 0 else 'red' for importance in feature_importances]
            ax.barh(simplified_feature_names, feature_importances, color=bar_colors, height=0.8)

            ax.set_xlabel('Feature Importance', fontsize=70)
            # Update the graph title to be more descriptive and user-friendly
            ax.set_title('Key Features Impacting This Prediction (LIME Explanation)', fontsize=80)
            ax.tick_params(axis='both', which='major', labelsize=60)
            # Rotate x-axis labels slightly to prevent overlapping
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout(pad=3.0)

            # Save the figure to a buffer and encode it as base64
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=200)  # High DPI for clarity
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            # Debugging: Ensure the image is properly encoded
            if not img_base64:
                print(f"Error: Failed to encode graph for model {model_name}")

            # Simplify feature names for the text explanation
            top_features = [feature for feature, importance in sorted(feature_importances_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]]

            # Generate a simplified, user-friendly explanation with only feature names
            explanation_text = (
                f"The model '{model_name.replace('_', ' ').title()}' predicted that the patient is '{result}'. "
                f"This conclusion was influenced by factors such as {', '.join(top_features)}. "
                f"The graph above highlights the most important features that contributed to this prediction."
            )

            # Store the results for this model
            results[model_name] = {
                "prediction": result,
                "confidence": confidence,
                "accuracy": accuracies[model_name],
                "lime_explanation": lime_explanation,
                "lime_explanation_image": img_base64,  # Ensure this is properly encoded
                "text_explanation": explanation_text,
            }
        
        # Return the results as a JSON response
        return jsonify(results)
    
    except Exception as e:
        # Handle errors gracefully and return an error message
        print(f"Error in predict function: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/tune', methods=['POST'])
def tune_models():
    """
    Handle POST requests to the /tune endpoint.
    This function performs hyperparameter tuning for the models using GridSearchCV.
    """
    try:
        # Define hyperparameter grids for each model
        param_grids = {
            "logistic_regression": {
                "C": [0.1, 1, 10],
                "solver": ["liblinear", "lbfgs"]
            },
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20]
            },
            "gradient_boosting": {
                "learning_rate": [0.01, 0.1, 0.2],
                "n_estimators": [50, 100, 200]
            }
        }

        results = {}

        # Iterate through the models and perform tuning
        for model_name, model in models.items():
            if model_name not in param_grids:
                continue

            param_grid = param_grids[model_name]
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, data['Outcome'])  # Assuming 'Outcome' is the target column

            # Perform cross-validation with the best parameters
            best_model = grid_search.best_estimator_
            cv_scores = cross_val_score(best_model, X_train, data['Outcome'], cv=5, scoring='accuracy')

            # Store the tuning results
            results[model_name] = {
                "best_params": grid_search.best_params_, 
                "cv_scores": cv_scores.tolist(),
                "mean_cv_score": np.mean(cv_scores)
            }

        # Return the tuning results as a JSON response
        return jsonify(results)

    except Exception as e:
        # Handle errors gracefully and return an error message
        print(f"Error in tune_models function: {e}")
        return jsonify({"error": f"Model tuning failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the Flask app in debug mode for easier development
    app.run(debug=True)