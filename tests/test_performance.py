
import time

import requests


API_URL = "http://localhost:8000/predict"

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


def test_api_performance():
    response_times = []

    for _ in range(10):
        start_time = time.perf_counter()

        response = requests.post(
            API_URL,
            json=sample_input,
            timeout=30
        )

        elapsed = (time.perf_counter() - start_time) * 1000

        assert response.status_code == 200
        response_times.append(elapsed)

    average_time = sum(response_times) / len(response_times)
    maximum_time = max(response_times)

    print(f"Average response time: {average_time:.2f} ms")
    print(f"Maximum response time: {maximum_time:.2f} ms")

    assert average_time < 5000
