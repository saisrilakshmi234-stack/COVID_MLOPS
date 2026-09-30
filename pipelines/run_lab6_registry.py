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
    print("COVID-19 MLOps - Lab 6 Model Registry")
    print("====================================")

    # Make sure execution happens from project root
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    os.chdir(project_root)

    # Step 1: Register model
    run_step(
        os.path.join("src", "train_registry.py"),
        "Model Registration"
    )

    # Step 2: Automate model lifecycle
    run_step(
        os.path.join("src", "automate_lifecycle.py"),
        "Model Lifecycle Automation"
    )

    # Step 3: Generate registry report
    run_step(
        os.path.join("src", "generate_registry_report.py"),
        "Registry Report Generation"
    )

    print("\n====================================")
    print("[SUCCESS] Lab 6 registry pipeline completed!")
    print("====================================")


if __name__ == "__main__":
    main()