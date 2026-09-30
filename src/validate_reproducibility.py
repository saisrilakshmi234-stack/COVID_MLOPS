import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# ==============================
# Configuration
# ==============================

RAW_DATA_PATH = "data/raw/country_wise.csv"
PROCESSED_DIR = "data/processed"

TARGET_COLUMN = "New_cases"

TEST_SIZE = 0.20
RANDOM_STATE = 42

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
# Load and prepare raw data
# ==============================

def load_and_prepare_data():

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    print(
        f"[INFO] Dataset loaded: {df.shape}"
    )

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(
            " ",
            "_",
            regex=False
        )
    )

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            f"not found."
        )

    # Remove rows with missing target
    df = df.dropna(
        subset=[TARGET_COLUMN]
    )

    y = df[TARGET_COLUMN]

    X = df.drop(
        columns=[TARGET_COLUMN]
    )

    # Remove unwanted columns
    columns_to_remove = [
        "Date",
        "SNo",
        "Unnamed:_0"
    ]

    for column in columns_to_remove:

        if column in X.columns:
            X = X.drop(
                columns=[column]
            )

    # Replace infinity values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return X, y


# ==============================
# Check reproducibility
# ==============================

def check_reproducibility():

    print(
        "\n[INFO] Checking reproducibility..."
    )

    # Load original processed outputs
    if not all(
        os.path.exists(path)
        for path in [
            X_TRAIN_PATH,
            X_TEST_PATH,
            Y_TRAIN_PATH,
            Y_TEST_PATH
        ]
    ):
        raise FileNotFoundError(
            "Processed pipeline output files "
            "are missing."
        )

    X_train_saved = np.load(
        X_TRAIN_PATH
    )

    X_test_saved = np.load(
        X_TEST_PATH
    )

    y_train_saved = np.load(
        Y_TRAIN_PATH
    )

    y_test_saved = np.load(
        Y_TEST_PATH
    )

    # Load raw data again
    X, y = load_and_prepare_data()

    # Repeat the exact same split
    X_train_new, X_test_new, y_train_new, y_test_new = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
    )

    print(
        "\n[INFO] Repeated split:"
    )

    print(
        f"X_train: {X_train_new.shape}"
    )

    print(
        f"X_test : {X_test_new.shape}"
    )

    print(
        f"y_train: {y_train_new.shape}"
    )

    print(
        f"y_test : {y_test_new.shape}"
    )

    # Check target reproducibility
    train_target_match = np.array_equal(
        y_train_saved,
        y_train_new.to_numpy()
    )

    test_target_match = np.array_equal(
        y_test_saved,
        y_test_new.to_numpy()
    )

    print(
        "\n[INFO] Target reproducibility:"
    )

    print(
        f"y_train match: "
        f"{train_target_match}"
    )

    print(
        f"y_test match : "
        f"{test_target_match}"
    )

    # Check split sizes
    split_shape_match = (
        X_train_saved.shape[0]
        == X_train_new.shape[0]
        and
        X_test_saved.shape[0]
        == X_test_new.shape[0]
    )

    print(
        "\n[INFO] Split size reproducibility:"
    )

    print(
        f"Split sizes match: "
        f"{split_shape_match}"
    )

    # Check expected dimensions
    feature_shape_match = (
        X_train_saved.shape[1] == 167
        and
        X_test_saved.shape[1] == 167
    )

    print(
        "\n[INFO] Feature dimension check:"
    )

    print(
        f"Expected 167 features: "
        f"{feature_shape_match}"
    )

    # Final validation
    if not train_target_match:
        raise ValueError(
            "Training target split is not "
            "reproducible."
        )

    if not test_target_match:
        raise ValueError(
            "Test target split is not "
            "reproducible."
        )

    if not split_shape_match:
        raise ValueError(
            "Train/test split sizes do not match."
        )

    if not feature_shape_match:
        raise ValueError(
            "Processed feature dimensions "
            "are not as expected."
        )

    print(
        "\n[SUCCESS] Reproducibility validation "
        "PASSED!"
    )


# ==============================
# Main
# ==============================

def main():

    print(
        "===================================="
    )

    print(
        "COVID-19 MLOps - "
        "Reproducibility Validation"
    )

    print(
        "===================================="
    )

    check_reproducibility()

    print(
        "\n===================================="
    )

    print(
        "[SUCCESS] ALL REPRODUCIBILITY "
        "CHECKS PASSED!"
    )

    print(
        "===================================="
    )


if __name__ == "__main__":
    main()