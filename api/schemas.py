from pydantic import BaseModel, Field


class COVIDPredictionRequest(BaseModel):
    country_region: str = Field(..., alias="Country/Region")
    confirmed: float = Field(..., alias="Confirmed", ge=0)
    deaths: float = Field(..., alias="Deaths", ge=0)
    recovered: float = Field(..., alias="Recovered", ge=0)
    active: float = Field(..., alias="Active", ge=0)
    new_deaths: float = Field(..., alias="New_deaths")
    new_recovered: float = Field(..., alias="New_recovered")
    deaths_per_100_cases: float = Field(..., alias="Deaths_/_100_Cases")
    recovered_per_100_cases: float = Field(..., alias="Recovered_/_100_Cases")
    deaths_per_100_recovered: float = Field(..., alias="Deaths_/_100_Recovered")
    confirmed_last_week: float = Field(..., alias="Confirmed_last_week", ge=0)
    one_week_change: float = Field(..., alias="1_week_change")
    one_week_percent_increase: float = Field(..., alias="1_week_%_increase")
    who_region: str = Field(..., alias="WHO_Region")

    model_config = {
        "populate_by_name": True,
        "extra": "forbid"
    }


class COVIDPredictionResponse(BaseModel):
    prediction: float
    target: str = "New_cases"
    model_name: str = "COVID_Ridge_Model"
    model_version: str
