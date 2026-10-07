import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
df = pd.read_csv('../data/heart.csv')

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'], drop_first=True)

# Separate features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Save the scaler
joblib.dump(scaler, '../models/scaler.pkl')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the processed data
X_train.to_csv('../data/X_train.csv', index=False)
X_test.to_csv('../data/X_test.csv', index=False)
y_train.to_csv('../data/y_train.csv', index=False)
y_test.to_csv('../data/y_test.csv', index=False)

print("Data preparation complete. Scaler and datasets saved.")
