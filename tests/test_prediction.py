from fastapi.testclient import TestClient

from api.main import app


sample_input = {
    "Country/Region": "India",
    "Confirmed": 100000,
    "Deaths": 1500,
    "Recovered": 95000,
    "Active": 3500,
    "New_deaths": 20,
    "New_recovered": 500,
    "Deaths_/_100_Cases": 1.5,
    "Recovered_/_100_Cases": 95.0,
    "Deaths_/_100_Recovered": 1.58,
    "Confirmed_last_week": 95000,
    "1_week_change": 5000,
    "1_week_%_increase": 5.26,
    "WHO_Region": "South-East Asia"
}


def test_prediction_endpoint():
    with TestClient(app) as client:
        response = client.post("/predict", json=sample_input)

        print("Prediction response:", response.json())

        assert response.status_code == 200
        assert "prediction" in response.json()
        assert response.json()["model_version"] == "6"
