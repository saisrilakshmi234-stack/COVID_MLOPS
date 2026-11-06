# LAB 13 - End-to-End MLOps Integration Report

## COVID-19 Prediction Project

### 1. Objective

To integrate and verify the COVID-19 prediction MLOps system, including model serving, Docker containerization, Kubernetes deployment, API testing, prediction consistency, performance testing, and inference logging.

### 2. Project Components

- Data validation and preprocessing.
- Ridge Regression model training and evaluation.
- MLflow experiment tracking and model registry.
- Production model export.
- FastAPI prediction service.
- Docker containerization.
- Kubernetes deployment and service management.
- API prediction logging and performance testing.

### 3. Model and Deployment Information

- Model Name: COVID_Ridge_Model
- Model Version: 6
- Model Type: Ridge Regression
- Prediction Target: New_cases
- API Framework: FastAPI
- Container Image: covid-mlops-api:lab13
- Docker Container: covid-lab13-demo
- Local API Port: 8000
- Docker API Port: 8002

### 4. API Integration Testing

Executed the project test suite using:

`python -m pytest .\tests\ -v -s`

Results:

- Total tests: 6
- Passed: 6
- Failed: 0
- Overall status: SUCCESS

Test cases:

1. Home endpoint verification.
2. Health endpoint verification.
3. API documentation verification.
4. Local versus Docker prediction consistency.
5. API performance testing.
6. Prediction endpoint verification.

### 5. Local and Docker Prediction Consistency

The same input data was submitted to the local API and Docker API.

- Local prediction: 2707.074755102518
- Docker prediction: 2707.074755102518
- Absolute difference: 0.0

Result: Both environments generated identical predictions.

### 6. API Performance Testing

Executed ten prediction requests against the local API.

- Average response time: 2100.88 ms
- Maximum response time: 2289.23 ms
- Average response time threshold: 5000 ms

Result: Performance test passed.

### 7. Inference Monitoring

Verified the inference log:

`logs/api_predictions.jsonl`

The log records:

- Timestamp
- Input request
- Prediction
- Inference latency
- Request status

Successful prediction records were verified.

Note: This confirms inference logging. Automated data drift detection and model-performance monitoring are not claimed as completed by this test.

### 8. Docker Verification

Built the Docker image:

`docker build -t covid-mlops-api:lab13 .`

Started the container:

`docker run -d --name covid-lab13-demo -p 8002:8000 covid-mlops-api:lab13`

Verified the health endpoint:

`http://127.0.0.1:8002/health`

Result: Docker container running and API health status healthy.

### 9. Kubernetes Integration

The Kubernetes deployment completed in Lab 12.

- Deployment: covid-api-deployment
- Service: covid-api-service
- Replicas: 3
- Kubernetes context: docker-desktop
- Service endpoint verification: Successful
- Scaling verification: Successful
- Self-healing verification: Successful

Refer to `LAB12_KUBERNETES_DEPLOYMENT_REPORT.md`.

### 10. Generated Reports

- `reports/api_testing_report.json`
- `reports/api_performance_report.json`
- `reports/deployment_automation_report.json`
- `logs/api_predictions.jsonl`

### 11. Warnings and Limitations

FastAPI currently uses the deprecated `@app.on_event("startup")` handler. Migration to lifespan event handlers is recommended.

This integration exercise verifies API functionality, container prediction consistency, performance, and inference logging. It does not independently establish automated drift detection or retraining.

### 12. Final Result

- API integration tests: Successful.
- Docker image build: Successful.
- Docker container startup: Successful.
- Docker health check: Successful.
- Local and Docker prediction consistency: Successful.
- API performance test: Successful.
- Inference logging: Verified.
- Kubernetes deployment: Completed in Lab 12.

**Status: LAB 13 INTEGRATION TESTING COMPLETED**

### 13. Conclusion

The COVID-19 MLOps project successfully integrates model serving, API testing, Docker deployment, Kubernetes orchestration, performance validation, and inference logging. The six-test suite passed, and local and containerized predictions matched exactly for the tested input.
