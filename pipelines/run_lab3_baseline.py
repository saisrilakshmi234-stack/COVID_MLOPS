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
    print("COVID-19 MLOps - Lab 3 Baseline")
    print("====================================")

    # Make sure the script is executed from project root
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    os.chdir(project_root)

    # Step 1: Preprocessing
    run_step(
        os.path.join("src", "preprocess.py"),
        "Data Preprocessing"
    )

    # Step 2: Model Training
    run_step(
        os.path.join("src", "train.py"),
        "Ridge Model Training"
    )

    # Step 3: Model Evaluation
    run_step(
        os.path.join("src", "evaluate.py"),
        "Model Evaluation"
    )

    print("\n====================================")
    print("[SUCCESS] Lab 3 baseline pipeline completed!")
    print("====================================")


if __name__ == "__main__":
    main()