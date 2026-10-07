import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]


def load_and_prepare_data():
    print("Loading and preparing data...")
    print("Current working directory:", os.getcwd())
    print("Contents of data directory:", os.listdir('data'))

    df = pd.read_csv('data/heart.csv', header=None, names=FEATURE_NAMES + ['target'])
    df = df.replace('?', pd.NA).apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['target'])
    df[FEATURE_NAMES] = df[FEATURE_NAMES].fillna(df[FEATURE_NAMES].median())

    X = df[FEATURE_NAMES]
    y = (df['target'] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURE_NAMES, index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURE_NAMES, index=X_test.index
    )

    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')

    return X_train, X_test, y_train, y_test


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    print("Training and evaluating models...")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True)
    }
    
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        positive_class_index = list(model.classes_).index(1)
        y_prob = model.predict_proba(X_test)[:, positive_class_index]

        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1-score': f1_score(y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_prob)
        }

        model_filename = f'models/{name.replace(" ", "_").lower()}.pkl'
        joblib.dump(model, model_filename)
        print(f"Saved {name} model to {model_filename}")

    results_df = pd.DataFrame(results).T
    print("\nModel Evaluation Results:")
    print(results_df)
    return results_df


def main():
    os.makedirs('models', exist_ok=True)
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    results.to_csv('models/model_results.csv')
    print("\nModel results saved to 'models/model_results.csv'")


if __name__ == "__main__":
    main()
