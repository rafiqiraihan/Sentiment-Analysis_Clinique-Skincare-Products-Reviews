import mlflow
import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def main(data_path):
    df = pd.read_csv(data_path)

    X = df['processed_text']
    y = df['sentiment']

    tfidf = TfidfVectorizer(max_features=200, min_df=17, max_df=0.8)
    X_tfidf = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    input_example = X_train[0:5]

    param_grid = {
        'C': [0.01, 0.2, 0.1, 1, 10],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'max_iter': [100, 200, 500, 700],
        'class_weight': ['balanced', None],
        'solver': ['liblinear', 'lbfgs', 'saga']  # <- perbaikan kecil, koma sebelumnya hilang
    }

    log_reg = LogisticRegression(random_state=42)

    grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)


    with mlflow.start_run():
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        mlflow.log_params(best_params)

        os.makedirs("model", exist_ok=True)
        joblib.dump(best_model, "model/model.pkl")
        mlflow.log_artifact("model/model.pkl")

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=input_example
        )

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="scripts/processed_reviews_clinique.csv", help="Path to the processed CSV dataset")
    args = parser.parse_args()
    main(args.data_path)