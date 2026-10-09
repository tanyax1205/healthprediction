"""Prediction request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

FeatureValue = Any  # int | float | str | bool | None


class PredictRequest(BaseModel):
    values: dict[str, FeatureValue] = Field(
        description="Feature name -> value. Omitted features are treated as "
        "missing (imputed), or 0 for binary-flag tasks like symptom screening."
    )


class ShapContribution(BaseModel):
    feature: str
    value: float | str | None = None
    contribution: float
    direction: str = Field(description="increases | decreases")
    sentence: str = ""


class OutlierWarning(BaseModel):
    feature: str
    value: float
    min: float | None = None
    max: float | None = None
    message: str


class TopClass(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    task_id: str
    label: str
    predicted: bool | None = None
    probability: float | None = Field(None, description="calibrated P(positive) or top-1")
    risk_level: str | None = Field(None, description="Low | Moderate | High (binary only)")
    threshold: float | None = None
    top_classes: list[TopClass]
    shap: list[ShapContribution]
    outlier_warnings: list[OutlierWarning]
    model_version: str
    best_model: str
    disclaimer: str