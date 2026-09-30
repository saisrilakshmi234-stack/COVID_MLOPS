import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


# ==============================
# Configuration
# ==============================

EXPERIMENT_NAME = "COVID-19-Ridge-Regression"
REGISTERED_MODEL_NAME = "COVID_Ridge_Model"

MODEL_PATH = os.path.join(
    "models",
    "ridge_model_mlflow.pkl"
)

METRICS_PATH = os.path.join(
    "outputs",
    "mlflow_evaluation_metrics.json"
)


# ==============================
# Find Latest MLflow Run
# ==============================

def get_latest_run():
    print("\n[INFO] Searching for latest MLflow run...")

    experiment = mlflow.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        raise ValueError(
            f"MLflow experiment not found: {EXPERIMENT_NAME}"
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if runs.empty:
        raise ValueError(
            "No MLflow runs found in the experiment."
        )

    run_id = runs.iloc[0]["run_id"]

    print(f"[SUCCESS] Latest run found: {run_id}")

    return run_id


# ==============================
# Register Model
# ==============================

def register_model(run_id):
    print("\n[INFO] Registering Ridge model...")

    model_uri = f"runs:/{run_id}/ridge_model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    print(
        f"[SUCCESS] Model registered successfully!"
    )

    print(
        f"[INFO] Model name: "
        f"{registered_model.name}"
    )

    print(
        f"[INFO] Model version: "
        f"{registered_model.version}"
    )

    return registered_model


# ==============================
# Add Model Description
# ==============================

def update_model_description():
    print("\n[INFO] Updating registered model description...")

    client = MlflowClient()

    client.update_registered_model(
        name=REGISTERED_MODEL_NAME,
        description=(
            "Ridge Regression model for predicting "
            "COVID-19 New cases using country-wise "
            "COVID-19 data."
        )
    )

    print(
        "[SUCCESS] Registered model description updated."
    )


# ==============================
# Main
# ==============================

def main():

    print("====================================")
    print("COVID-19 MLOps - Model Registry")
    print("====================================")

    # Make sure MLflow uses the local tracking database
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    # Get latest training run
    run_id = get_latest_run()

    # Register model
    registered_model = register_model(run_id)

    # Update model description
    update_model_description()

    print("\n====================================")
    print("[SUCCESS] Lab 6 Model Registry completed!")
    print("====================================")

    print(
        f"\nRegistered Model : "
        f"{registered_model.name}"
    )

    print(
        f"Model Version    : "
        f"{registered_model.version}"
    )

    print(
        f"Run ID           : "
        f"{run_id}"
    )


if __name__ == "__main__":
    main()