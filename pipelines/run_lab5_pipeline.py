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
    print("COVID-19 MLOps - Lab 5 Pipeline")
    print("====================================")

    # Make sure execution happens from project root
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    os.chdir(project_root)

    # Step 1: Build preprocessing pipeline
    run_step(
        os.path.join("src", "preprocess_pipeline.py"),
        "Preprocessing Pipeline"
    )

    # Step 2: Validate data
    run_step(
        os.path.join("src", "validate_data.py"),
        "Data Validation"
    )

    # Step 3: Validate reproducibility
    run_step(
        os.path.join("src", "validate_reproducibility.py"),
        "Reproducibility Validation"
    )

    # Step 4: Validate generated outputs
    run_step(
        os.path.join("src", "validate_outputs.py"),
        "Output Validation"
    )

    print("\n====================================")
    print("[SUCCESS] Lab 5 pipeline completed!")
    print("====================================")


if __name__ == "__main__":
    main()