import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ============================================================
# Configuration
# ============================================================

RAW_DATA_PATH = "data/raw/country_wise.csv"
PROCESSED_DIR = "data/processed"

TARGET_COLUMN = "New cases"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# Load Dataset
# ============================================================

def load_data(path=RAW_DATA_PATH):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}"
        )

    df = pd.read_csv(path)

    print(f"[INFO] Dataset loaded: {df.shape}")

    return df


# ============================================================
# Clean Column Names
# ============================================================

def clean_column_names(df):

    df = df.copy()

    # Remove leading/trailing spaces
    df.columns = df.columns.str.strip()

    # Replace spaces with underscores
    df.columns = df.columns.str.replace(
        " ",
        "_",
        regex=False
    )

    return df


# ============================================================
# Prepare Features and Target
# ============================================================

def prepare_data(df):

    df = df.copy()

    # After cleaning column names:
    # "New cases" becomes "New_cases"
    target = TARGET_COLUMN.replace(" ", "_")

    if target not in df.columns:

        raise ValueError(
            f"Target column '{target}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # Remove rows where target is missing
    df = df.dropna(subset=[target])

    # Target
    y = df[target]

    # Features
    X = df.drop(columns=[target])

    # Remove columns that should not be used
    columns_to_remove = [
        "Date",
        "SNo",
        "Unnamed:_0"
    ]

    for column in columns_to_remove:

        if column in X.columns:
            X = X.drop(columns=[column])

    return X, y


# ============================================================
# Build Preprocessing Pipeline
# ============================================================

def build_preprocessor(X):

    numerical_columns = X.select_dtypes(
        include=[
            "int64",
            "float64",
            "int32",
            "float32"
        ]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=[
            "object",
            "category",
            "bool"
        ]
    ).columns.tolist()

    print("\n[INFO] Numerical columns:")
    print(numerical_columns)

    print("\n[INFO] Categorical columns:")
    print(categorical_columns)

    # -----------------------------
    # Numerical preprocessing
    # -----------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # -----------------------------
    # Categorical preprocessing
    # -----------------------------

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


# ============================================================
# Main Preprocessing Function
# ============================================================

def preprocess_data(
    input_path=RAW_DATA_PATH,
    output_dir=PROCESSED_DIR
):

    # Create output directory
    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    df = load_data(input_path)

    # --------------------------------------------------------
    # 2. Clean column names
    # --------------------------------------------------------

    df = clean_column_names(df)

    print("\n[INFO] Columns:")
    print(df.columns.tolist())

    # --------------------------------------------------------
    # 3. Prepare X and y
    # --------------------------------------------------------

    X, y = prepare_data(df)

    # --------------------------------------------------------
    # 4. Handle infinity values
    # --------------------------------------------------------

    # Some COVID percentage/ratio columns can contain
    # infinity when their denominator is zero.

    infinity_count = np.isinf(
        X.select_dtypes(
            include=np.number
        )
    ).sum().sum()

    print(
        f"\n[INFO] Infinity values found: "
        f"{infinity_count}"
    )

    # Convert +inf and -inf to NaN
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    print(
        "[INFO] Infinity values replaced "
        "with NaN"
    )

    # --------------------------------------------------------
    # 5. Display shapes
    # --------------------------------------------------------

    print(
        f"\n[INFO] Feature shape: {X.shape}"
    )

    print(
        f"[INFO] Target shape: {y.shape}"
    )

    # --------------------------------------------------------
    # 6. Train/Test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
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

    # --------------------------------------------------------
    # 7. Build preprocessing pipeline
    # --------------------------------------------------------

    preprocessor = build_preprocessor(
        X_train
    )

    # --------------------------------------------------------
    # 8. Fit and transform training data
    # --------------------------------------------------------

    X_train_processed = (
        preprocessor.fit_transform(
            X_train
        )
    )

    # --------------------------------------------------------
    # 9. Transform test data
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 10. Save processed data
    # --------------------------------------------------------

    np.save(
        os.path.join(
            output_dir,
            "X_train.npy"
        ),
        X_train_processed
    )

    np.save(
        os.path.join(
            output_dir,
            "X_test.npy"
        ),
        X_test_processed
    )

    np.save(
        os.path.join(
            output_dir,
            "y_train.npy"
        ),
        y_train.to_numpy()
    )

    np.save(
        os.path.join(
            output_dir,
            "y_test.npy"
        ),
        y_test.to_numpy()
    )

    # --------------------------------------------------------
    # 11. Save preprocessing pipeline
    # --------------------------------------------------------

    pipeline_path = os.path.join(
        output_dir,
        "preprocessor.pkl"
    )

    joblib.dump(
        preprocessor,
        pipeline_path
    )

    print(
        f"\n[SUCCESS] Preprocessor saved to: "
        f"{pipeline_path}"
    )

    print("\n[SUCCESS] Processed files saved:")
    print(
        " - data/processed/X_train.npy"
    )
    print(
        " - data/processed/X_test.npy"
    )
    print(
        " - data/processed/y_train.npy"
    )
    print(
        " - data/processed/y_test.npy"
    )
    print(
        " - data/processed/preprocessor.pkl"
    )

    return (
        X_train_processed,
        X_test_processed,
        y_train.to_numpy(),
        y_test.to_numpy(),
        preprocessor
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    preprocess_data()

    print(
        "\n[SUCCESS] Preprocessing "
        "completed successfully!"
    )