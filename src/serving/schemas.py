"""Pydantic request / response models for the serving API."""

from __future__ import annotations

from pydantic import BaseModel, Field


# -- Requests ----------------------------------------------------------------


class PredictRequest(BaseModel):
    """Single-sample prediction request."""

    features: dict[str, float] = Field(..., description="Feature name → value mapping")


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""

    samples: list[dict[str, float]] = Field(
        ..., description="List of feature name → value mappings"
    )


# -- Responses ---------------------------------------------------------------


class PredictResponse(BaseModel):
    """Single-sample prediction response."""

    predicted_class: int
    predicted_label: str
    probabilities: dict[str, float]


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[PredictResponse]


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    model_type: str
    features: list[str]
    n_classes: int
    class_names: list[str]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    model_loaded: bool
