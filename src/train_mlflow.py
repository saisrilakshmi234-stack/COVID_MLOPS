import os
import json
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

X_TRAIN_PATH = os.path.join(PROCESSED_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(PROCESSED_DIR, "y_train.npy")
X_TEST_PATH = os.path.join(PROCESSED_DIR, "X_test.npy")
Y_TEST_PATH = os.path.join(PROCESSED_DIR, "y_test.npy")

MODEL_PATH = os.path.join(MODEL_DIR, "ridge_model_mlflow.pkl")
METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "mlflow_evaluation_metrics.json"
)

MLFLOW_EXPERIMENT = "COVID-19-Ridge-Regression"

RANDOM_STATE = 42
ALPHA = 1.0


def load_data():
    print("[INFO] Loading processed data...")

    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape : {X_test.shape}")
    print(f"y_test shape : {y_test.shape}")

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    print("\n[INFO] Training Ridge Regression...")

    model = Ridge(
        alpha=ALPHA,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    print("[SUCCESS] Model training completed")

    return model


def evaluate_model(model, X_test, y_test):
    print("\n[INFO] Evaluating model...")

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2)
    }

    print("\n========== Evaluation Results ==========")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return metrics


def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"\n[SUCCESS] Model saved to: "
        f"{MODEL_PATH}"
    )


def save_metrics(metrics):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(
        METRICS_PATH,
        "w"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    print(
        f"[SUCCESS] Metrics saved to: "
        f"{METRICS_PATH}"
    )


def main():

    print("====================================")
    print("COVID-19 MLOps - MLflow Tracking")
    print("====================================")

    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Set MLflow experiment
    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    with mlflow.start_run():

        print("\n[INFO] MLflow run started")

        # Log parameters
        mlflow.log_param(
            "model",
            "Ridge Regression"
        )

        mlflow.log_param(
            "alpha",
            ALPHA
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE
        )

        mlflow.log_param(
            "train_samples",
            X_train.shape[0]
        )

        mlflow.log_param(
            "test_samples",
            X_test.shape[0]
        )

        mlflow.log_param(
            "features",
            X_train.shape[1]
        )

        # Train
        model = train_model(
            X_train,
            y_train
        )

        # Evaluate
        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        # Log metrics
        mlflow.log_metric(
            "MAE",
            metrics["MAE"]
        )

        mlflow.log_metric(
            "MSE",
            metrics["MSE"]
        )

        mlflow.log_metric(
            "RMSE",
            metrics["RMSE"]
        )

        mlflow.log_metric(
            "R2",
            metrics["R2"]
        )

        # Log model
        mlflow.sklearn.log_model(
            model,
            name="ridge_model"
        )

        print(
            "\n[SUCCESS] Model logged to MLflow"
        )

        # Save local model and metrics
        save_model(model)
        save_metrics(metrics)

        print(
            "\n[INFO] MLflow run completed"
        )


if __name__ == "__main__":
    main()