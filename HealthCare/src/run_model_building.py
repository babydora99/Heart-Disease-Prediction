import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

# Create models directory if it doesn't exist
os.makedirs('../models', exist_ok=True)

# Load the preprocessed data
try:
    X_train = pd.read_csv('../data/X_train.csv')
    X_test = pd.read_csv('../data/X_test.csv')
    y_train = pd.read_csv('../data/y_train.csv').values.ravel()
    y_test = pd.read_csv('../data/y_test.csv').values.ravel()
    print("Data loaded successfully!")
except Exception as e:
    print(f"Error loading data: {e}")
    print("Please run prepare_data.py first to generate the required data files.")
    exit(1)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    results[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    
    # Save the trained model
    joblib.dump(model, f'models/{name.replace(" ", "_")}.pkl')

# Display the results
results_df = pd.DataFrame(results).T
print(results_df)
