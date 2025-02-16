import numpy as np
import pandas as pd
import torch
import json
import os
import joblib  # For saving models
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =========================== CONFIGURATION =========================== #
N_TEST = 1000000
INPUT_PATH = "/dccstor/aml_datasets/bis/"
OUTPUT_PATH = "/dccstor/aml_datasets/bis/models/"

if N_TEST is not None:
    INPUT_PATH = f"/dccstor/aml_datasets/bis_{N_TEST}k/"
    OUTPUT_PATH = f"/dccstor/aml_datasets/bis_{N_TEST}k/models"
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    logging.info(f"Output path set to: {OUTPUT_PATH}")
    

# =========================== LOAD CONFIGURATION =========================== #
def load_config(config_file="config_ml.json"):
    with open(config_file, "r") as json_file:
        config = json.load(json_file)
    return config["training_parameters"]

# =========================== LOAD DATA =========================== #
def load_data(x_train_path, x_test_path, y_train_path, y_test_path):
    X_train = torch.load(x_train_path)
    X_test = torch.load(x_test_path)
    y_train = torch.load(y_train_path)
    y_test = torch.load(y_test_path)
    return X_train, X_test, y_train, y_test

# =========================== DATA PREPROCESSING =========================== #
def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

# =========================== TRAINING MODELS =========================== #
def train_models(X_train, y_train, parameters):
    logging.info("Training models...")
    
    models = {
        "Logistic Regression": LogisticRegression(**parameters["Logistic Regression"]),
        "Random Forest": RandomForestClassifier(**parameters["Random Forest"]),
        "Gradient Boosting": GradientBoostingClassifier(**parameters["Gradient Boosting"]),
        "XGBoost": XGBClassifier(**parameters["XGBoost"]),
        "LightGBM": LGBMClassifier(**parameters["LightGBM"])
    }

    # Ensemble (Voting Classifier)
    ensemble = VotingClassifier(
        estimators=[
            ("XGBoost", models["XGBoost"]),
            ("LightGBM", models["LightGBM"]),
            ("Random Forest", models["Random Forest"]),
            ("Gradient Boosting", models["Gradient Boosting"]),
        ],
        voting="soft"
    )
    models["Ensemble"] = ensemble

    for name, model in models.items():
        logging.info(f"Training {name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, f"{OUTPUT_PATH}_{name.replace(' ', '_').lower()}.pkl")
    
    return models

# =========================== MODEL EVALUATION =========================== #
def evaluate_models(models, X_test, y_test):
    results = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="binary"),
            "Recall": recall_score(y_test, y_pred, average="binary"),
            "F1-score": f1_score(y_test, y_pred, average="binary"),
        }
        results[name] = metrics
        logging.info(f"{name} - F1 Score: {metrics['F1-score']:.4f}")

    results_df = pd.DataFrame(results).T
    logging.info("Model evaluation completed.")
    return results_df

# =========================== MAIN FUNCTION =========================== #
def main():
    logging.info("Starting training pipeline...")
    
    # Load training parameters from JSON
    parameters = load_config()

    # Load Data
    X_train, X_test, y_train, y_test = load_data(
        os.path.join(INPUT_PATH, f"X_train_{N_TEST if N_TEST is not None else ''}.pt"),
        os.path.join(INPUT_PATH, f"X_test_{N_TEST if N_TEST is not None else ''}.pt"),
        os.path.join(INPUT_PATH, f"y_train_{N_TEST if N_TEST is not None else ''}.pt"),
        os.path.join(INPUT_PATH, f"y_test_{N_TEST if N_TEST is not None else ''}.pt")
    )
    
    # Preprocess Data
    X_train, X_test, _ = preprocess_data(X_train, X_test)

    # Train Models
    models = train_models(X_train, y_train[:,0], parameters)

    # Evaluate Models
    results_df = evaluate_models(models, X_test, y_test[:,0])
    results_df.to_csv(OUTPUT_PATH+"model_results.csv")
    logging.info("✅ Model results saved to model_results.csv")

if __name__ == "__main__":
    main()
