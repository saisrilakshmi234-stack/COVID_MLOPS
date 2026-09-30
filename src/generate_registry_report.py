import json
import mlflow
from mlflow.tracking import MlflowClient


# ==============================
# Configuration
# ==============================
REGISTERED_MODEL_NAME = "COVID_Ridge_Model"
METRICS_PATH = "outputs/mlflow_evaluation_metrics.json"
REPORT_PATH = "outputs/registry_report.json"

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()


def main():
    print("====================================")
    print("COVID-19 MLOps - Registry Report")
    print("====================================")

    # --------------------------------
    # Step 1: Load evaluation metrics
    # --------------------------------
    print("\n[INFO] Loading evaluation metrics...")

    with open(METRICS_PATH, "r") as file:
        metrics = json.load(file)

    print("[SUCCESS] Evaluation metrics loaded.")

    # --------------------------------
    # Step 2: Find Production model
    # --------------------------------
    print("\n[INFO] Reading model registry information...")

    versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    production_versions = [
        version
        for version in versions
        if version.current_stage == "Production"
    ]

    if not production_versions:
        print("[ERROR] No Production model version found.")
        return

    production_version = max(
        production_versions,
        key=lambda x: int(x.version)
    )

    model_version = int(production_version.version)

    print(f"[INFO] Model Name : {production_version.name}")
    print(f"[INFO] Version    : {production_version.version}")
    print(f"[INFO] Stage      : {production_version.current_stage}")
    print(f"[INFO] Status     : {production_version.status}")

    # --------------------------------
    # Step 3: Generate report
    # --------------------------------
    print("\n[INFO] Generating registry report...")

    report = {
        "registered_model": production_version.name,
        "model_description": client.get_registered_model(
            REGISTERED_MODEL_NAME
        ).description,
        "model_version": model_version,
        "stage": production_version.current_stage,
        "status": production_version.status,
        "run_id": production_version.run_id,
        "source": production_version.source,
        "version_description": production_version.description,
        "created_timestamp": production_version.creation_timestamp,
        "last_updated_timestamp": production_version.last_updated_timestamp,
        "metrics": {
            "MAE": metrics.get("MAE"),
            "MSE": metrics.get("MSE"),
            "RMSE": metrics.get("RMSE"),
            "R2": metrics.get("R2")
        }
    }

    with open(REPORT_PATH, "w") as file:
        json.dump(report, file, indent=4)

    print(f"[SUCCESS] Registry report saved to: {REPORT_PATH}")

    # --------------------------------
    # Step 4: Display report
    # --------------------------------
    print("\n====================================")
    print("MODEL REGISTRY REPORT")
    print("====================================")

    print(f"Registered Model : {report['registered_model']}")
    print(f"Model Version    : {report['model_version']}")
    print(f"Stage            : {report['stage']}")
    print(f"Status           : {report['status']}")
    print(f"Run ID           : {report['run_id']}")

    print("\nEvaluation Metrics:")
    print(f"MAE  : {report['metrics']['MAE']:.4f}")
    print(f"MSE  : {report['metrics']['MSE']:.4f}")
    print(f"RMSE : {report['metrics']['RMSE']:.4f}")
    print(f"R²   : {report['metrics']['R2']:.4f}")

    print("\n====================================")
    print("[SUCCESS] Registry report generation completed!")
    print("====================================")


if __name__ == "__main__":
    main()