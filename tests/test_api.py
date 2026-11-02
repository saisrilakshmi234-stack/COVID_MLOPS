from fastapi.testclient import TestClient

from api.main import app


def test_home_endpoint():
    with TestClient(app) as client:
        response = client.get("/")

        assert response.status_code == 200
        assert response.json()["message"] == "COVID-19 Prediction API is running"


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


def test_api_documentation():
    with TestClient(app) as client:
        response = client.get("/docs")

        assert response.status_code == 200
