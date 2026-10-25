from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    headline: str = Field(min_length=1, max_length=1000)
    article: str = Field(min_length=1, max_length=100000)
    model: str = Field(min_length=1)


class Prediction(BaseModel):
    prediction: Literal["REAL", "FAKE", "UNCERTAIN"]
    confidence: float
    real_probability: float
    fake_probability: float
    model: str
    explanation_note: str


class ErrorResponse(BaseModel):
    error: str
    code: str
