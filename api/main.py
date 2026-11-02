import time

from fastapi import FastAPI, HTTPException

from api.model_loader import load_artifacts
from api.preprocessing import preprocess_request
from api.monitoring_logger import log_prediction
from api.schemas import COVIDPredictionRequest, COVIDPredictionResponse


app = FastAPI(
    title="COVID-19 Prediction API",
    description="API for predicting new COVID-19 cases using the production Ridge model.",
    version="1.0.0"
)

model = None
preprocessor = None
metadata = {}


@app.on_event("startup")
def startup_event():
    global model, preprocessor, metadata
    model, preprocessor, metadata = load_artifacts()


@app.get("/")
def home():
    return {
        "message": "COVID-19 Prediction API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are not loaded"
        )

    return {
        "status": "healthy",
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version")
    }


@app.post("/predict", response_model=COVIDPredictionResponse)
def predict(request: COVIDPredictionRequest):
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are not loaded"
        )

    start_time = time.perf_counter()

    try:
        processed_data = preprocess_request(request, preprocessor)
        prediction = model.predict(processed_data)[0]
        latency_ms = (time.perf_counter() - start_time) * 1000

        log_prediction(
            request.model_dump(by_alias=True),
            prediction,
            latency_ms
        )

        return COVIDPredictionResponse(
            prediction=float(prediction),
            model_name=metadata.get("model_name", "COVID_Ridge_Model"),
            model_version=str(metadata.get("model_version", "unknown"))
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        ) from error
