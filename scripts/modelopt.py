import mlflow
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Set tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Set Experiment
mlflow.set_experiment("Analisis Sentiment_Random Forest_GridSearch")

# Load dataset
df = pd.read_csv("scripts/processed_reviews_clinique.csv")

# Split dataset into text features and sentiment labels
X = df['processed_text']
y = df['sentiment']

# Extract text features using TF-IDF
tfidf = TfidfVectorizer(max_features=200, min_df=17, max_df=0.8)
X_tfidf = tfidf.fit_transform(X)

# Convert extracted features to a DataFrame
features_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())

# Split dataset into training and test 
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

input_example = X_train[0:5]

# Parameter grid
param_grid = {
    'C': [0.01, 0.2, 0.1, 1, 10],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'max_iter': [100, 200, 500, 700],
    'class_weight': ['balanced', None],
    'solver': ['liblinear', 'lbfgs' 'saga']
}

# Model
log_reg = LogisticRegression(random_state=42)

# Grid Search
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Log Model
with mlflow.start_run():
    # Best Model from GridSearch
    model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Logging Best Model
    mlflow.log_params(best_params)

    # Save model
    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl")
    mlflow.sklearn.log_model(model, "model", input_example=input_example)

    # Prediction and evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)