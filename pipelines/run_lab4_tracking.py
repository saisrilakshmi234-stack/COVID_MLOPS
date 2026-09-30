import os
import subprocess
import sys


def run_step(script_path, step_name):
    print("\n====================================")
    print(f"Running: {step_name}")
    print("====================================")

    result = subprocess.run(
        [sys.executable, script_path],
        check=False
    )

    if result.returncode != 0:
        print(
            f"\n[ERROR] {step_name} failed "
            f"with exit code {result.returncode}"
        )
        sys.exit(result.returncode)

    print(f"[SUCCESS] {step_name} completed.")


def main():
    print("====================================")
    print("COVID-19 MLOps - Lab 4 MLflow Tracking")
    print("====================================")

    # Make sure execution happens from project root
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    os.chdir(project_root)

    # Run MLflow tracking and model logging
    run_step(
        os.path.join("src", "train_mlflow.py"),
        "MLflow Experiment Tracking"
    )

    print("\n====================================")
    print("[SUCCESS] Lab 4 MLflow pipeline completed!")
    print("====================================")


if __name__ == "__main__":
    main()