import os
import numpy as np
import joblib

from sklearn.linear_model import Ridge


# -----------------------------
# Paths
# -----------------------------
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"

X_TRAIN_PATH = os.path.join(PROCESSED_DIR, "X_train.npy")
Y_TRAIN_PATH = os.path.join(PROCESSED_DIR, "y_train.npy")

MODEL_PATH = os.path.join(MODEL_DIR, "ridge_model.pkl")


# -----------------------------
# Model configuration
# -----------------------------
RANDOM_STATE = 42
ALPHA = 1.0


def load_processed_data():
    """Load preprocessed training data."""

    if not os.path.exists(X_TRAIN_PATH):
        raise FileNotFoundError(
            f"Training features not found: {X_TRAIN_PATH}"
        )

    if not os.path.exists(Y_TRAIN_PATH):
        raise FileNotFoundError(
            f"Training target not found: {Y_TRAIN_PATH}"
        )

    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH)

    print("[INFO] Training data loaded")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    return X_train, y_train


def train_model(X_train, y_train):
    """Train Ridge Regression model."""

    print("\n[INFO] Training Ridge Regression model...")

    model = Ridge(
        alpha=ALPHA,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    print("[SUCCESS] Model training completed")

    return model


def save_model(model):
    """Save trained model."""

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    print(f"[SUCCESS] Model saved to: {MODEL_PATH}")


def main():
    print("====================================")
    print("COVID-19 MLOps - Model Training")
    print("====================================")

    # 1. Load processed data
    X_train, y_train = load_processed_data()

    # 2. Train model
    model = train_model(X_train, y_train)

    # 3. Save model
    save_model(model)

    print("\n====================================")
    print("Training completed successfully!")
    print("====================================")


if __name__ == "__main__":
    main()