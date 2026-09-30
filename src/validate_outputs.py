import os
import json
import numpy as np
import joblib


# ==============================
# Configuration
# ==============================

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

# Lab 3 outputs
LAB3_FILES = {
    "X_train": os.path.join(
        PROCESSED_DIR,
        "X_train.npy"
    ),
    "X_test": os.path.join(
        PROCESSED_DIR,
        "X_test.npy"
    ),
    "y_train": os.path.join(
        PROCESSED_DIR,
        "y_train.npy"
    ),
    "y_test": os.path.join(
        PROCESSED_DIR,
        "y_test.npy"
    ),
    "preprocessor": os.path.join(
        PROCESSED_DIR,
        "preprocessor.pkl"
    ),
    "model": os.path.join(
        MODEL_DIR,
        "ridge_model.pkl"
    ),
    "metrics": os.path.join(
        OUTPUT_DIR,
        "evaluation_metrics.json"
    )
}

# Lab 4 MLflow output
MLFLOW_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "ridge_model_mlflow.pkl"
)

MLFLOW_METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "mlflow_evaluation_metrics.json"
)

# Lab 5 pipeline outputs
PIPELINE_FILES = {
    "pipeline_X_train": os.path.join(
        PROCESSED_DIR,
        "pipeline_X_train.npy"
    ),
    "pipeline_X_test": os.path.join(
        PROCESSED_DIR,
        "pipeline_X_test.npy"
    ),
    "pipeline_y_train": os.path.join(
        PROCESSED_DIR,
        "pipeline_y_train.npy"
    ),
    "pipeline_y_test": os.path.join(
        PROCESSED_DIR,
        "pipeline_y_test.npy"
    ),
    "preprocessing_pipeline": os.path.join(
        PROCESSED_DIR,
        "preprocessing_pipeline.pkl"
    )
}


# ==============================
# Check File Exists
# ==============================

def check_file(path, name):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"{name} not found: {path}"
        )

    print(
        f"[OK] {name}: {path}"
    )


# ==============================
# Validate Numpy Arrays
# ==============================

def validate_array(
    path,
    name,
    expected_shape
):

    check_file(
        path,
        name
    )

    array = np.load(path)

    print(
        f"[INFO] {name} shape: "
        f"{array.shape}"
    )

    if array.shape != expected_shape:

        raise ValueError(
            f"{name} has unexpected "
            f"shape {array.shape}. "
            f"Expected {expected_shape}."
        )

    if np.isnan(array).any():

        raise ValueError(
            f"{name} contains NaN values."
        )

    if np.isinf(array).any():

        raise ValueError(
            f"{name} contains infinity values."
        )

    print(
        f"[OK] {name} validation passed."
    )

    return array


# ==============================
# Validate Model
# ==============================

def validate_model():

    print(
        "\n[INFO] Validating trained models..."
    )

    check_file(
        LAB3_FILES["model"],
        "Lab 3 Ridge model"
    )

    check_file(
        MLFLOW_MODEL_PATH,
        "MLflow Ridge model"
    )

    model = joblib.load(
        LAB3_FILES["model"]
    )

    mlflow_model = joblib.load(
        MLFLOW_MODEL_PATH
    )

    print(
        f"[INFO] Lab 3 model type: "
        f"{type(model).__name__}"
    )

    print(
        f"[INFO] MLflow model type: "
        f"{type(mlflow_model).__name__}"
    )

    if type(model).__name__ != "Ridge":

        raise ValueError(
            "Lab 3 model is not Ridge."
        )

    if type(mlflow_model).__name__ != "Ridge":

        raise ValueError(
            "MLflow model is not Ridge."
        )

    print(
        "[SUCCESS] Model validation passed."
    )


# ==============================
# Validate Metrics
# ==============================

def validate_metrics(
    path,
    name
):

    check_file(
        path,
        name
    )

    with open(
        path,
        "r"
    ) as file:

        metrics = json.load(file)

    required_metrics = [
        "MAE",
        "MSE",
        "RMSE",
        "R2"
    ]

    for metric in required_metrics:

        if metric not in metrics:

            raise ValueError(
                f"{metric} missing from "
                f"{name}."
            )

        if not isinstance(
            metrics[metric],
            (int, float)
        ):

            raise ValueError(
                f"{metric} in {name} "
                f"is not numeric."
            )

    print(
        f"[INFO] {name}:"
    )

    print(
        f"  MAE  : {metrics['MAE']:.4f}"
    )

    print(
        f"  MSE  : {metrics['MSE']:.4f}"
    )

    print(
        f"  RMSE : {metrics['RMSE']:.4f}"
    )

    print(
        f"  R2   : {metrics['R2']:.4f}"
    )

    print(
        f"[OK] {name} validation passed."
    )

    return metrics


# ==============================
# Validate Pipeline
# ==============================

def validate_pipeline():

    print(
        "\n[INFO] Validating Lab 5 pipeline..."
    )

    check_file(
        PIPELINE_FILES[
            "preprocessing_pipeline"
        ],
        "Preprocessing pipeline"
    )

    pipeline = joblib.load(
        PIPELINE_FILES[
            "preprocessing_pipeline"
        ]
    )

    if pipeline is None:

        raise ValueError(
            "Preprocessing pipeline "
            "could not be loaded."
        )

    print(
        f"[INFO] Pipeline type: "
        f"{type(pipeline).__name__}"
    )

    print(
        "[SUCCESS] Preprocessing pipeline "
        "validation passed."
    )


# ==============================
# Main
# ==============================

def main():

    print(
        "===================================="
    )

    print(
        "COVID-19 MLOps - Output Validation"
    )

    print(
        "===================================="
    )

    # --------------------------------
    # Validate Lab 3 arrays
    # --------------------------------

    print(
        "\n[INFO] Validating Lab 3 outputs..."
    )

    validate_array(
        LAB3_FILES["X_train"],
        "X_train",
        (149, 167)
    )

    validate_array(
        LAB3_FILES["X_test"],
        "X_test",
        (38, 167)
    )

    validate_array(
        LAB3_FILES["y_train"],
        "y_train",
        (149,)
    )

    validate_array(
        LAB3_FILES["y_test"],
        "y_test",
        (38,)
    )

    check_file(
        LAB3_FILES["preprocessor"],
        "Lab 3 preprocessor"
    )

    # --------------------------------
    # Validate Lab 3 model
    # --------------------------------

    validate_model()

    # --------------------------------
    # Validate Lab 3 metrics
    # --------------------------------

    print(
        "\n[INFO] Validating Lab 3 metrics..."
    )

    validate_metrics(
        LAB3_FILES["metrics"],
        "Lab 3 evaluation metrics"
    )

    # --------------------------------
    # Validate MLflow metrics
    # --------------------------------

    print(
        "\n[INFO] Validating MLflow outputs..."
    )

    validate_metrics(
        MLFLOW_METRICS_PATH,
        "MLflow evaluation metrics"
    )

    # --------------------------------
    # Validate Lab 5 pipeline
    # --------------------------------

    validate_pipeline()

    # --------------------------------
    # Validate Lab 5 arrays
    # --------------------------------

    print(
        "\n[INFO] Validating Lab 5 outputs..."
    )

    validate_array(
        PIPELINE_FILES[
            "pipeline_X_train"
        ],
        "Pipeline X_train",
        (149, 167)
    )

    validate_array(
        PIPELINE_FILES[
            "pipeline_X_test"
        ],
        "Pipeline X_test",
        (38, 167)
    )

    validate_array(
        PIPELINE_FILES[
            "pipeline_y_train"
        ],
        "Pipeline y_train",
        (149,)
    )

    validate_array(
        PIPELINE_FILES[
            "pipeline_y_test"
        ],
        "Pipeline y_test",
        (38,)
    )

    # --------------------------------
    # Final result
    # --------------------------------

    print(
        "\n===================================="
    )

    print(
        "[SUCCESS] ALL OUTPUT "
        "VALIDATIONS PASSED!"
    )

    print(
        "===================================="
    )


if __name__ == "__main__":

    main()