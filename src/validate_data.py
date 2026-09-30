import os
import numpy as np
import pandas as pd


# ==============================
# Configuration
# ==============================

RAW_DATA_PATH = "data/raw/country_wise.csv"
PROCESSED_DIR = "data/processed"

TARGET_COLUMN = "New_cases"

X_TRAIN_PATH = os.path.join(
    PROCESSED_DIR,
    "pipeline_X_train.npy"
)

X_TEST_PATH = os.path.join(
    PROCESSED_DIR,
    "pipeline_X_test.npy"
)

Y_TRAIN_PATH = os.path.join(
    PROCESSED_DIR,
    "pipeline_y_train.npy"
)

Y_TEST_PATH = os.path.join(
    PROCESSED_DIR,
    "pipeline_y_test.npy"
)


# ==============================
# Validate Raw Dataset
# ==============================

def validate_raw_dataset():

    print("\n[INFO] Validating raw dataset...")

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Raw dataset not found: "
            f"{RAW_DATA_PATH}"
        )

    df = pd.read_csv(
        RAW_DATA_PATH
    )

    print(
        f"[INFO] Dataset shape: "
        f"{df.shape}"
    )

    # Check target column
    if TARGET_COLUMN not in (
        df.columns
        .str.strip()
        .str.replace(
            " ",
            "_",
            regex=False
        )
        .tolist()
    ):
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            f"not found."
        )

    # Check empty dataset
    if df.empty:
        raise ValueError(
            "Dataset is empty."
        )

    # Check duplicate rows
    duplicate_count = df.duplicated().sum()

    print(
        f"[INFO] Duplicate rows: "
        f"{duplicate_count}"
    )

    # Check missing values
    missing_values = df.isna().sum().sum()

    print(
        f"[INFO] Missing values: "
        f"{missing_values}"
    )

    # Check infinity values
    numeric_df = df.select_dtypes(
        include=np.number
    )

    infinity_count = np.isinf(
        numeric_df
    ).sum().sum()

    print(
        f"[INFO] Infinity values: "
        f"{infinity_count}"
    )

    print(
        "[SUCCESS] Raw dataset validation "
        "completed."
    )

    return True


# ==============================
# Validate Processed Data
# ==============================

def validate_processed_data():

    print(
        "\n[INFO] Validating processed data..."
    )

    required_files = [
        X_TRAIN_PATH,
        X_TEST_PATH,
        Y_TRAIN_PATH,
        Y_TEST_PATH
    ]

    for file_path in required_files:

        if not os.path.exists(file_path):

            raise FileNotFoundError(
                f"Required processed file "
                f"not found: {file_path}"
            )

    # Load processed arrays
    X_train = np.load(
        X_TRAIN_PATH
    )

    X_test = np.load(
        X_TEST_PATH
    )

    y_train = np.load(
        Y_TRAIN_PATH
    )

    y_test = np.load(
        Y_TEST_PATH
    )

    print(
        f"[INFO] X_train shape: "
        f"{X_train.shape}"
    )

    print(
        f"[INFO] X_test shape: "
        f"{X_test.shape}"
    )

    print(
        f"[INFO] y_train shape: "
        f"{y_train.shape}"
    )

    print(
        f"[INFO] y_test shape: "
        f"{y_test.shape}"
    )

    # Check sample counts
    if X_train.shape[0] != y_train.shape[0]:

        raise ValueError(
            "X_train and y_train "
            "sample counts do not match."
        )

    if X_test.shape[0] != y_test.shape[0]:

        raise ValueError(
            "X_test and y_test "
            "sample counts do not match."
        )

    # Check NaN values
    if np.isnan(X_train).any():
        raise ValueError(
            "NaN values found in X_train."
        )

    if np.isnan(X_test).any():
        raise ValueError(
            "NaN values found in X_test."
        )

    if np.isnan(y_train).any():
        raise ValueError(
            "NaN values found in y_train."
        )

    if np.isnan(y_test).any():
        raise ValueError(
            "NaN values found in y_test."
        )

    # Check infinity values
    if np.isinf(X_train).any():
        raise ValueError(
            "Infinity values found "
            "in X_train."
        )

    if np.isinf(X_test).any():
        raise ValueError(
            "Infinity values found "
            "in X_test."
        )

    if np.isinf(y_train).any():
        raise ValueError(
            "Infinity values found "
            "in y_train."
        )

    if np.isinf(y_test).any():
        raise ValueError(
            "Infinity values found "
            "in y_test."
        )

    # Check feature dimensions
    if X_train.ndim != 2:
        raise ValueError(
            "X_train must be a 2D array."
        )

    if X_test.ndim != 2:
        raise ValueError(
            "X_test must be a 2D array."
        )

    # Check train/test feature consistency
    if X_train.shape[1] != X_test.shape[1]:

        raise ValueError(
            "X_train and X_test have "
            "different numbers of features."
        )

    print(
        f"[INFO] Number of features: "
        f"{X_train.shape[1]}"
    )

    print(
        "[SUCCESS] Processed data validation "
        "completed."
    )

    return True


# ==============================
# Main Validation
# ==============================

def main():

    print(
        "===================================="
    )

    print(
        "COVID-19 MLOps - Data Validation"
    )

    print(
        "===================================="
    )

    try:

        validate_raw_dataset()

        validate_processed_data()

        print(
            "\n===================================="
        )

        print(
            "[SUCCESS] ALL DATA VALIDATIONS "
            "PASSED!"
        )

        print(
            "===================================="
        )

    except Exception as error:

        print(
            "\n[ERROR] Data validation failed:"
        )

        print(error)

        raise


if __name__ == "__main__":

    main()