
import json
import os
import shutil
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient


MODEL_NAME = "COVID_Ridge_Model"
TRACKING_URI = "sqlite:///mlflow.db"

SOURCE_PREPROCESSOR = "data/processed/preprocessing_pipeline.pkl"
DATA_PATH = "data/raw/country_wise.csv"

DEPLOYMENT_DIR = "deployment"
REPORT_DIR = "reports"

MODEL_PATH = os.path.join(DEPLOYMENT_DIR, "model.pkl")
PREPROCESSOR_PATH = os.path.join(
    DEPLOYMENT_DIR, "preprocessor.pkl"
)
METADATA_PATH = os.path.join(
    DEPLOYMENT_DIR, "model_metadata.json"
)
REPORT_PATH = os.path.join(
    REPORT_DIR, "deployment_automation_report.json"
)


def deploy_model():
    os.makedirs(DEPLOYMENT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    print("\n[INFO] Checking Production model...")

    versions = client.get_latest_versions(
        MODEL_NAME,
        stages=["Production"]
    )

    if not versions:
        raise RuntimeError(
            f"No Production version found for {MODEL_NAME}"
        )

    production_version = versions[0]
    version = str(production_version.version)

    print(f"[INFO] Model: {MODEL_NAME}")
    print(f"[INFO] Version: {version}")
    print("[INFO] Loading model from MLflow...")

    model_uri = f"models:/{MODEL_NAME}/{version}"
    model = mlflow.sklearn.load_model(model_uri)

    if not os.path.exists(SOURCE_PREPROCESSOR):
        raise FileNotFoundError(
            f"Preprocessor not found: {SOURCE_PREPROCESSOR}"
        )

    preprocessor = joblib.load(SOURCE_PREPROCESSOR)

    print("[INFO] Checking preprocessing compatibility...")

    df = pd.read_csv(DATA_PATH)
    df.columns = (
        df.columns.str.strip().str.replace(" ", "_", regex=False)
    )

    if "New_cases" not in df.columns:
        raise ValueError("New_cases target column is missing")

    X = df.drop(columns=["New_cases"])
    X = X.drop(
        columns=[
            col for col in ["Date", "SNo", "Unnamed:_0"]
            if col in X.columns
        ]
    )

    X = X.replace([np.inf, -np.inf], np.nan)

    sample = X.head(5)
    transformed = preprocessor.transform(sample)

    predictions = model.predict(transformed)

    if len(predictions) != len(sample):
        raise RuntimeError("Prediction validation failed")

    print("[SUCCESS] Sample predictions generated")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    metadata = {
        "project": "COVID-19 MLOps",
        "model_name": MODEL_NAME,
        "model_version": version,
        "model_stage": production_version.current_stage,
        "model_type": "Ridge Regression",
        "target": "New_cases",
        "preprocessing": "Saved training preprocessing pipeline",
        "deployed_at": datetime.now(timezone.utc).isoformat(),
        "model_path": MODEL_PATH,
        "preprocessor_path": PREPROCESSOR_PATH
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    report = {
        "status": "SUCCESS",
        "model_name": MODEL_NAME,
        "model_version": version,
        "stage": production_version.current_stage,
        "model_exported": os.path.exists(MODEL_PATH),
        "preprocessor_exported": os.path.exists(
            PREPROCESSOR_PATH
        ),
        "metadata_created": os.path.exists(METADATA_PATH),
        "sample_predictions_validated": True,
        "sample_prediction_count": len(predictions)
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n===================================")
    print("[SUCCESS] COVID MODEL DEPLOYED")
    print("===================================")
    print(f"Model: {MODEL_PATH}")
    print(f"Preprocessor: {PREPROCESSOR_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    deploy_model()
