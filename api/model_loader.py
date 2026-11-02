import json
from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
DEPLOYMENT_DIR = BASE_DIR / "deployment"

MODEL_PATH = DEPLOYMENT_DIR / "model.pkl"
PREPROCESSOR_PATH = DEPLOYMENT_DIR / "preprocessor.pkl"
METADATA_PATH = DEPLOYMENT_DIR / "model_metadata.json"


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return model, preprocessor, metadata
