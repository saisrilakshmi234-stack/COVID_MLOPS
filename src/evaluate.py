import os
import json
import numpy as np
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# -----------------------------
# Paths
# -----------------------------
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

X_TEST_PATH = os.path.join(PROCESSED_DIR, "X_test.npy")
Y_TEST_PATH = os.path.join(PROCESSED_DIR, "y_test.npy")

MODEL_PATH = os.path.join(MODEL_DIR, "ridge_model.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")


def load_test_data():
    """Load processed test data."""

    if not os.path.exists(X_TEST_PATH):
        raise FileNotFoundError(
            f"Test features not found: {X_TEST_PATH}"
        )

    if not os.path.exists(Y_TEST_PATH):
        raise FileNotFoundError(
            f"Test target not found: {Y_TEST_PATH}"
        )

    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    print("[INFO] Test data loaded")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")

    return X_test, y_test


def load_model():
    """Load the trained Ridge Regression model."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print("[INFO] Trained model loaded")
    print(f"Model: {type(model).__name__}")

    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model using regression metrics."""

    print("\n[INFO] Generating predictions...")

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    return metrics


def save_metrics(metrics):
    """Save evaluation metrics as JSON."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"\n[SUCCESS] Metrics saved to: {METRICS_PATH}")


def main():

    print("====================================")
    print("COVID-19 MLOps - Model Evaluation")
    print("====================================")

    # 1. Load test data
    X_test, y_test = load_test_data()

    # 2. Load trained model
    model = load_model()

    # 3. Evaluate model
    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    # 4. Display results
    print("\n========== Evaluation Results ==========")

    print(f"MAE  : {metrics['MAE']:.4f}")
    print(f"MSE  : {metrics['MSE']:.4f}")
    print(f"RMSE : {metrics['RMSE']:.4f}")
    print(f"R²   : {metrics['R2']:.4f}")

    # 5. Save results
    save_metrics(metrics)

    print("\n====================================")
    print("Evaluation completed successfully!")
    print("====================================")


if __name__ == "__main__":
    main()