# LAB 11 – Docker Containerization Report

## 1. Objective
The objective of Lab 11 was to package and deploy the COVID-19 prediction API using Docker.

## 2. API Implementation
The FastAPI application provides:
- GET / : API welcome endpoint
- GET /health : Model health-check endpoint
- POST /predict : COVID-19 new-case prediction endpoint

The API loads the exported Ridge model and preprocessing pipeline from the deployment directory.

## 3. Docker Configuration
- Base image: python:3.13-slim
- Working directory: /app
- Application port: 8000
- Container image: covid-mlops-api:latest
- Container name: covid-container

## 4. Containerization Activities
1. Created the Dockerfile.
2. Created the .dockerignore file.
3. Built the Docker image successfully.
4. Started the container with port mapping 8000:8000.
5. Verified the API documentation and health endpoint.
6. Sent a prediction request to the running container.
7. Verified prediction monitoring logs.

## 5. Kubernetes Configuration
Created the following Kubernetes resource files:
- k8s/deployment.yaml – Deployment with 2 replicas.
- k8s/service.yaml – ClusterIP service on port 8000.

These configuration files were created but not applied to a Kubernetes cluster.

## 6. Testing Results
- API endpoint tests: PASS
- Prediction endpoint test: PASS
- Local versus Docker prediction test: PASS
- Performance test: PASS
- Combined test suite: 6 passed, 0 failed

## 7. Performance Results
- Number of requests: 10
- Average response time: 41.04 ms
- Maximum response time: 94.16 ms

## 8. Conclusion
The COVID-19 prediction API was successfully containerized and tested using Docker. The container returned valid predictions, and local and container predictions matched. Kubernetes deployment configuration files were also prepared.
