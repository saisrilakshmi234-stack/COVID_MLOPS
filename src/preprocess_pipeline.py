import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ==============================
# Configuration
# ==============================

RAW_DATA_PATH = "data/raw/country_wise.csv"
PROCESSED_DIR = "data/processed"

TARGET_COLUMN = "New_cases"

TEST_SIZE = 0.20
RANDOM_STATE = 42

PIPELINE_PATH = os.path.join(
    PROCESSED_DIR,
    "preprocessing_pipeline.pkl"
)


# ==============================
# Load Dataset
# ==============================

def load_data(path=RAW_DATA_PATH):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    print(
        f"[INFO] Dataset loaded: {df.shape}"
    )

    return df


# ==============================
# Clean Column Names
# ==============================

def clean_column_names(df):

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(
            " ",
            "_",
            regex=False
        )
    )

    return df


# ==============================
# Prepare Features and Target
# ==============================

def prepare_data(df):

    df = df.copy()

    target = TARGET_COLUMN

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' "
            f"not found.\n"
            f"Available columns: "
            f"{df.columns.tolist()}"
        )

    # Remove rows where target is missing
    df = df.dropna(
        subset=[target]
    )

    y = df[target]

    X = df.drop(
        columns=[target]
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

    return X, y


# ==============================
# Build Preprocessing Pipeline
# ==============================

def build_preprocessor(X):

    numerical_columns = (
        X.select_dtypes(
            include=[
                "int64",
                "float64",
                "int32",
                "float32"
            ]
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        X.select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        )
        .columns
        .tolist()
    )

    print("\n[INFO] Numerical columns:")
    print(numerical_columns)

    print("\n[INFO] Categorical columns:")
    print(categorical_columns)

    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    transformers = []

    if numerical_columns:

        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_columns
            )
        )

    if categorical_columns:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    return preprocessor


# ==============================
# Main Pipeline
# ==============================

def run_preprocessing():

    print(
        "===================================="
    )
    print(
        "COVID-19 MLOps - Reproducible "
        "Preprocessing Pipeline"
    )
    print(
        "===================================="
    )

    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True
    )

    # Load data
    df = load_data()

    # Clean column names
    df = clean_column_names(df)

    print("\n[INFO] Columns:")
    print(df.columns.tolist())

    # Prepare data
    X, y = prepare_data(df)

    # Handle infinity values
    numerical_data = X.select_dtypes(
        include=np.number
    )

    infinity_count = np.isinf(
        numerical_data
    ).sum().sum()

    print(
        f"\n[INFO] Infinity values found: "
        f"{infinity_count}"
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    print(
        "[INFO] Infinity values replaced "
        "with NaN"
    )

    print(
        f"\n[INFO] Feature shape: {X.shape}"
    )

    print(
        f"[INFO] Target shape: {y.shape}"
    )

    # Train/Test split
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
    )

    print("\n[INFO] Train/Test split:")
    print(
        f"X_train: {X_train.shape}"
    )
    print(
        f"X_test : {X_test.shape}"
    )
    print(
        f"y_train: {y_train.shape}"
    )
    print(
        f"y_test : {y_test.shape}"
    )

    # Build preprocessing pipeline
    preprocessor = build_preprocessor(
        X_train
    )

    # Fit only on training data
    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    # Transform test data
    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    print("\n[INFO] Processed data:")
    print(
        f"X_train_processed: "
        f"{X_train_processed.shape}"
    )
    print(
        f"X_test_processed : "
        f"{X_test_processed.shape}"
    )

    # Save processed data
    np.save(
        os.path.join(
            PROCESSED_DIR,
            "pipeline_X_train.npy"
        ),
        X_train_processed
    )

    np.save(
        os.path.join(
            PROCESSED_DIR,
            "pipeline_X_test.npy"
        ),
        X_test_processed
    )

    np.save(
        os.path.join(
            PROCESSED_DIR,
            "pipeline_y_train.npy"
        ),
        y_train.to_numpy()
    )

    np.save(
        os.path.join(
            PROCESSED_DIR,
            "pipeline_y_test.npy"
        ),
        y_test.to_numpy()
    )

    # Save complete preprocessing pipeline
    joblib.dump(
        preprocessor,
        PIPELINE_PATH
    )

    print(
        f"\n[SUCCESS] Preprocessing pipeline "
        f"saved to:"
    )

    print(
        f" {PIPELINE_PATH}"
    )

    print("\n[SUCCESS] Pipeline output files:")

    print(
        " - data/processed/"
        "pipeline_X_train.npy"
    )

    print(
        " - data/processed/"
        "pipeline_X_test.npy"
    )

    print(
        " - data/processed/"
        "pipeline_y_train.npy"
    )

    print(
        " - data/processed/"
        "pipeline_y_test.npy"
    )

    print(
        " - data/processed/"
        "preprocessing_pipeline.pkl"
    )

    print(
        "\n===================================="
    )

    print(
        "Preprocessing pipeline completed!"
    )

    print(
        "===================================="
    )


# ==============================
# Run
# ==============================

if __name__ == "__main__":

    run_preprocessing()