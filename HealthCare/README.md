# Heart Disease Prediction

A small Flask application that demonstrates binary heart-disease classification using five scikit-learn models. It is an educational project, not a medical diagnostic tool.

## What the prediction means

The project uses `data/heart.csv`, a headerless dataset with 13 patient features and a target in its final column. Training maps the original target to two classes:

- `0`: no heart disease
- `1`–`4`: heart disease

The app shows each model's predicted class and its estimated probability for the positive class (heart disease). The models can disagree, so there is no single guaranteed output. A probability such as `100%` is the model's estimate, not certainty that a person has the condition. Do not use these results to make medical decisions.

## Requirements

- Windows with Python installed
- PowerShell

## Install

Run these commands from the project folder:

```powershell
cd "C:\Users\lenovo\OneDrive\Desktop\HealthCare"
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `.venv` already exists, skip the `py -m venv .venv` command.

## Train the models

Training reads the dataset, creates a train/test split, evaluates the models, and saves the fitted models, scaler, and metrics under `models\`.

```powershell
.\.venv\Scripts\python.exe run_healthcare_project.py
```

Train the models before starting the app, and retrain whenever you change the dataset or training code.

## Run the web app

```powershell
.\.venv\Scripts\python.exe app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser. Keep the PowerShell window open while using the app; press **Ctrl+C** in that window to stop it.

## Models and evaluation

The project trains Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM) classifiers. Evaluation metrics are written to `models\model_results.csv`. Some models use randomness during training, so results can vary between training runs.

## Input features

The prediction form collects age, sex, chest-pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG result, maximum heart rate, exercise-induced angina, ST depression (`oldpeak`), slope, number of major vessels (`ca`), and thalassemia (`thal`). Categorical values are checked against the values available in the dataset.

## Project files

- `app.py`: Flask routes and prediction handling
- `run_healthcare_project.py`: data preparation, model training, and evaluation
- `data\heart.csv`: input dataset
- `models\`: trained models, scaler, and evaluation results
- `templates\index.html`: prediction form and results display

## Technologies

Python, Flask, Pandas, NumPy, and scikit-learn.
