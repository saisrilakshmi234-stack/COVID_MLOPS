
import requests
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


def test_local_vs_container_prediction():
    # Get prediction from local API
    with TestClient(app) as client:
        local_response = client.post(
            "/predict",
            json=sample_input
        )

    assert local_response.status_code == 200
    local_prediction = local_response.json()["prediction"]

    # Get prediction from Docker container
    docker_response = requests.post(
        "http://localhost:8002/predict",
        json=sample_input,
        timeout=30
    )

    assert docker_response.status_code == 200
    docker_prediction = docker_response.json()["prediction"]

    print("Local prediction:", local_prediction)
    print("Docker prediction:", docker_prediction)

    assert abs(local_prediction - docker_prediction) < 0.000001
