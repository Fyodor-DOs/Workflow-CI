"""
Modelling script untuk MLflow Project (Kriteria 3).
Script ini melatih RandomForestClassifier dengan MLflow autolog.
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
    # Load preprocessed dataset
    data_path = os.path.join(os.path.dirname(__file__), "..", "iris_preprocessing", "iris_preprocessed.csv")
    df = pd.read_csv(data_path)
    
    # Pisahkan fitur dan target
    X = df.drop(columns=['species_encoded'])
    y = df['species_encoded']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # MLflow experiment
    mlflow.set_experiment("Iris_Classification_CI")
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="RandomForest_CI"):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

if __name__ == "__main__":
    main()
