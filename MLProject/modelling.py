"""
Modelling for MLflow Project (Kriteria 3) - Titanic.
Run via: mlflow run MLProject --env-manager=local
"""
import os, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow, mlflow.sklearn

def main():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "titanic_preprocessed.csv")
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape}")
    X = df.drop(columns=['survived'])
    y = df['survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    mlflow.sklearn.autolog()
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_accuracy", accuracy)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

if __name__ == "__main__":
    main()
