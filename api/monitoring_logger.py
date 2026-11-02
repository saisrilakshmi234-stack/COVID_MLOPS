import json
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "api_predictions.jsonl"


def log_prediction(request_data, prediction, latency_ms):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request": request_data,
        "prediction": float(prediction),
        "latency_ms": float(latency_ms),
        "status": "success"
    }

    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(json.dumps(log_entry) + "\n")
