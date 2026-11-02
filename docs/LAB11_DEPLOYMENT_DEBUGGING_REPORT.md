# LAB 11 – Deployment Debugging Report

## 1. Objective
This report documents the issues encountered while deploying and testing the COVID-19 prediction API and the solutions applied.

## 2. Issues and Solutions

### Issue 1: Scikit-learn Version Mismatch
Problem:
The exported model initially produced a warning because the scikit-learn version used to load it differed from the version used to train or serialize it.

Solution:
Installed scikit-learn version 1.9.0 and regenerated the model and preprocessing artifacts.

Result:
Model and preprocessor loaded successfully.

### Issue 2: Docker Desktop Not Running
Problem:
Docker commands initially could not connect to the Docker engine.

Solution:
Started Docker Desktop and verified the engine using docker info.

Result:
Docker commands worked successfully.

### Issue 3: FastAPI Test Startup
Problem:
The health endpoint test initially failed because the application startup event had not run in the test client.

Solution:
Used FastAPI TestClient as a context manager.

Result:
The application startup event ran and the health endpoint test passed.

### Issue 4: Test Client Dependency
Problem:
The test environment required an additional HTTP client dependency.

Solution:
Installed the required HTTP client package.

Result:
FastAPI tests ran successfully.

## 3. Final Verification
- Model loading: PASS
- Preprocessor loading: PASS
- Docker image build: PASS
- Docker container startup: PASS
- Health endpoint: PASS
- Prediction endpoint: PASS
- Local versus Docker prediction comparison: PASS
- Performance test: PASS
- Combined test suite: 6 passed, 0 failed

## 4. Remaining Warning
FastAPI reports a deprecation warning for the older @app.on_event("startup") method. It does not prevent the API or tests from working. A lifespan event handler can replace it in a future improvement.

## 5. Conclusion
The deployment issues encountered during the project were resolved. The COVID-19 prediction API runs successfully in Docker, produces valid predictions, and passes all six automated tests.
