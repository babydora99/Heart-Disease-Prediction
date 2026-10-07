import pandas as pd
import joblib

# Load the trained model and the scaler
model = joblib.load('../models/Logistic_Regression.pkl')
scaler = joblib.load('../models/scaler.pkl')

# Load the training data columns to ensure consistency
training_cols = pd.read_csv('../data/X_train.csv').columns

def predict_heart_disease(input_data):
    """
    Preprocesses user input, makes a prediction, and returns the result.
    """
    # Create a DataFrame from the user input
    df = pd.DataFrame([input_data])

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'], drop_first=True)

    # Reindex the DataFrame to match the training data columns
    df = df.reindex(columns=training_cols, fill_value=0)

    # Scale numerical features
    numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    # Make a prediction
    prediction = model.predict(df)
    probability = model.predict_proba(df)[:, 1]

    return prediction[0], probability[0]

if __name__ == '__main__':
    # Example patient data (replace with actual input)
    patient_data = {
        'age': 52,
        'sex': 1,  # 1 for male, 0 for female
        'cp': 1,   # Chest pain type
        'trestbps': 125,
        'chol': 212,
        'fbs': 0,  # Fasting blood sugar
        'restecg': 1,
        'thalach': 168,
        'exang': 0,  # Exercise induced angina
        'oldpeak': 1.0,
        'slope': 2,
        'ca': 2,   # Number of major vessels
        'thal': 3
    }

    prediction, probability = predict_heart_disease(patient_data)

    print("--- Heart Disease Prediction ---")
    if prediction == 1:
        print(f"Prediction: The patient is likely to have heart disease.")
        print(f"Probability: {probability:.2f}")
    else:
        print(f"Prediction: The patient is unlikely to have heart disease.")
        print(f"Probability of having heart disease: {probability:.2f}")
