import mlflow
from mlflow.tracking import MlflowClient


# ==============================
# Configuration
# ==============================
REGISTERED_MODEL_NAME = "COVID_Ridge_Model"

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()


def main():
    print("====================================")
    print("COVID-19 MLOps - Lifecycle Automation")
    print("====================================")

    # --------------------------------
    # Step 1: Find latest model version
    # --------------------------------
    print("\n[INFO] Checking registered model...")

    versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    if not versions:
        print("[ERROR] No registered model versions found.")
        return

    versions = sorted(
        versions,
        key=lambda x: int(x.version),
        reverse=True
    )

    latest_version = versions[0]
    model_version = int(latest_version.version)

    print(
        f"[INFO] Latest Model: {REGISTERED_MODEL_NAME} "
        f"| Version: {model_version} "
        f"| Stage: {latest_version.current_stage}"
    )

    # --------------------------------
    # Step 2: Update description
    # --------------------------------
    print("\n[INFO] Updating model version description...")

    client.update_model_version(
        name=REGISTERED_MODEL_NAME,
        version=str(model_version),
        description=(
            "Latest COVID-19 Ridge Regression model "
            "automatically registered and promoted."
        )
    )

    print("[SUCCESS] Model version description updated.")

    # --------------------------------
    # Step 3: Move latest version to Production
    # --------------------------------
    print("\n[INFO] Transitioning latest model to Production...")

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=str(model_version),
        stage="Production",
        archive_existing_versions=True
    )

    print("[SUCCESS] Latest model version transitioned to Production.")

    # --------------------------------
    # Step 4: Verify
    # --------------------------------
    print("\n[INFO] Verifying model lifecycle state...")

    updated_version = client.get_model_version(
        name=REGISTERED_MODEL_NAME,
        version=str(model_version)
    )

    print(f"[INFO] Model Name : {updated_version.name}")
    print(f"[INFO] Version    : {updated_version.version}")
    print(f"[INFO] Stage      : {updated_version.current_stage}")
    print(f"[INFO] Status     : {updated_version.status}")

    print("\n[SUCCESS] Model lifecycle verification completed.")

    print("\n====================================")
    print("[SUCCESS] Model lifecycle automation completed!")
    print("====================================")


if __name__ == "__main__":
    main()