import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt


# ============================================
# COVID-19 MLOps - Artifact Generation
# ============================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
X_TEST_PATH = os.path.join(
    BASE_DIR, "data", "processed", "X_test.npy"
)

Y_TEST_PATH = os.path.join(
    BASE_DIR, "data", "processed", "y_test.npy"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "ridge_model.pkl"
)

METRICS_PATH = os.path.join(
    BASE_DIR, "outputs", "evaluation_metrics.json"
)

ARTIFACTS_DIR = os.path.join(
    BASE_DIR, "artifacts"
)

# Create artifacts directory if it does not exist
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


print("====================================")
print("COVID-19 MLOps - Artifact Generation")
print("====================================")


# ============================================
# 1. Load Test Data
# ============================================

print("[INFO] Loading test data...")

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print(f"[INFO] X_test shape: {X_test.shape}")
print(f"[INFO] y_test shape: {y_test.shape}")


# ============================================
# 2. Load Ridge Model
# ============================================

print("[INFO] Loading Ridge model...")

# Model was saved using pickle/joblib-compatible format.
# joblib.load() correctly handles this sklearn model file.
model = joblib.load(MODEL_PATH)

print("[INFO] Ridge model loaded successfully.")


# ============================================
# 3. Generate Predictions
# ============================================

print("[INFO] Generating predictions...")

y_pred = model.predict(X_test)

print("[INFO] Predictions generated successfully.")


# ============================================
# 4. Load Evaluation Metrics
# ============================================

print("[INFO] Loading evaluation metrics...")

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

print("[INFO] Evaluation metrics loaded.")


# ============================================
# 5. Actual vs Predicted Plot
# ============================================

print("[INFO] Creating actual_vs_predicted.png...")

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

# Perfect prediction reference line
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual New Cases")
plt.ylabel("Predicted New Cases")
plt.title("Actual vs Predicted - Ridge Regression")
plt.grid(True, alpha=0.3)

plt.tight_layout()

actual_predicted_path = os.path.join(
    ARTIFACTS_DIR,
    "actual_vs_predicted.png"
)

plt.savefig(
    actual_predicted_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"[INFO] Saved: {actual_predicted_path}"
)


# ============================================
# 6. Residual Plot
# ============================================

print("[INFO] Creating residual_plot.png...")

residuals = y_test - y_pred

plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted New Cases")
plt.ylabel("Residuals")
plt.title("Residual Plot - Ridge Regression")
plt.grid(True, alpha=0.3)

plt.tight_layout()

residual_path = os.path.join(
    ARTIFACTS_DIR,
    "residual_plot.png"
)

plt.savefig(
    residual_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"[INFO] Saved: {residual_path}"
)


# ============================================
# 7. Prediction Error Distribution
# ============================================

print("[INFO] Creating prediction_error_distribution.png...")

errors = y_test - y_pred

plt.figure(figsize=(8, 6))

plt.hist(
    errors,
    bins=15,
    edgecolor="black"
)

plt.axvline(
    x=0,
    linestyle="--"
)

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Prediction Error Distribution")
plt.grid(True, alpha=0.3)

plt.tight_layout()

error_distribution_path = os.path.join(
    ARTIFACTS_DIR,
    "prediction_error_distribution.png"
)

plt.savefig(
    error_distribution_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"[INFO] Saved: {error_distribution_path}"
)


# ============================================
# 8. Model Performance Report
# ============================================

print("[INFO] Creating model_performance_report.json...")

model_performance_report = {
    "model": "Ridge Regression",
    "target": "New_cases",
    "test_samples": int(len(y_test)),
    "test_features": int(X_test.shape[1]),
    "metrics": {
        "MAE": metrics.get("MAE"),
        "MSE": metrics.get("MSE"),
        "RMSE": metrics.get("RMSE"),
        "R2": metrics.get("R2")
    }
}

performance_report_path = os.path.join(
    ARTIFACTS_DIR,
    "model_performance_report.json"
)

with open(performance_report_path, "w") as f:
    json.dump(
        model_performance_report,
        f,
        indent=4
    )

print(
    f"[INFO] Saved: {performance_report_path}"
)


# ============================================
# 9. Summary
# ============================================

print()
print("====================================")
print("Artifact Generation Completed")
print("====================================")

print("[INFO] Generated artifacts:")

print(" - actual_vs_predicted.png")
print(" - residual_plot.png")
print(" - prediction_error_distribution.png")
print(" - model_performance_report.json")

print()
print("[INFO] Artifact directory:")
print(ARTIFACTS_DIR)

print()
print("SUCCESS")