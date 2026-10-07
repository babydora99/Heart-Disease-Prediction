from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
feature_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]
categorical_values = {
    'sex': {0, 1},
    'cp': {1, 2, 3, 4},
    'fbs': {0, 1},
    'restecg': {0, 1, 2},
    'exang': {0, 1},
    'slope': {1, 2, 3},
    'ca': {0, 1, 2, 3},
    'thal': {3, 6, 7}
}

models = {
    'Logistic Regression': joblib.load(MODELS_DIR / 'logistic_regression.pkl'),
    'Decision Tree': joblib.load(MODELS_DIR / 'decision_tree.pkl'),
    'Random Forest': joblib.load(MODELS_DIR / 'random_forest.pkl'),
    'KNN': joblib.load(MODELS_DIR / 'knn.pkl'),
    'SVM': joblib.load(MODELS_DIR / 'svm.pkl')
}
scaler = joblib.load(MODELS_DIR / 'scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    missing_features = [feature for feature in feature_names if feature not in request.form]
    if missing_features:
        return jsonify({
            'status': 'error',
            'message': f"Missing required fields: {', '.join(missing_features)}"
        }), 400

    try:
        input_data = {
            feature: float(request.form[feature])
            for feature in feature_names
        }
    except ValueError:
        return jsonify({
            'status': 'error',
            'message': 'All feature values must be numeric.'
        }), 400

    if not np.isfinite(list(input_data.values())).all():
        return jsonify({
            'status': 'error',
            'message': 'Feature values must be finite numbers.'
        }), 400

    invalid_categories = [
        feature for feature, valid_values in categorical_values.items()
        if input_data[feature] not in valid_values
    ]
    if invalid_categories:
        return jsonify({
            'status': 'error',
            'message': (
                'Invalid category value(s): '
                f"{', '.join(invalid_categories)}. Use values from the form."
            )
        }), 400

    input_frame = pd.DataFrame([input_data], columns=feature_names)
    scaled_data = pd.DataFrame(
        scaler.transform(input_frame), columns=feature_names
    )

    predictions = {}
    for name, model in models.items():
        prediction = int(model.predict(scaled_data)[0])
        positive_class_index = list(model.classes_).index(1)
        probability = float(
            model.predict_proba(scaled_data)[0][positive_class_index]
        )
        predictions[name] = {
            'prediction': prediction,
            'probability': probability
        }

    return jsonify({
        'status': 'success',
        'predictions': predictions,
        'input_data': input_data
    })

if __name__ == '__main__':
    app.run(debug=False)
