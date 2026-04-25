"""
Modelling script untuk MLflow Project (Kriteria 3).
Dijalankan via: mlflow run MLProject --env-manager=local

Script ini melatih RandomForestClassifier dengan MLflow autolog.
Dataset harus berada di folder yang sama (iris_preprocessed.csv).
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

def main():
    # Load preprocessed dataset (relative to MLProject folder)
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iris_preprocessed.csv")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape}")

    # Pisahkan fitur dan target
    X = df.drop(columns=['species_encoded'])
    y = df['species_encoded']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Enable MLflow autolog
    mlflow.sklearn.autolog()

    # Train model (mlflow run already creates an active run)
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log additional metrics manually
    mlflow.log_metric("test_accuracy", accuracy)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    print("Model training completed and logged to MLflow!")

if __name__ == "__main__":
    main()
