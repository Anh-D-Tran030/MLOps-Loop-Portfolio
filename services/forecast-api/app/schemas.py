"""Pydantic v2 request/response schemas for the Forecast API."""

from pydantic import BaseModel


class PredictRequest(BaseModel):
    store_nbr: int
    family: str
    onpromotion: int
    days_ahead: int = 30


class PredictResponse(BaseModel):
    forecast: list[float]
    model_version: str
    prediction_id: str  # uuid4
