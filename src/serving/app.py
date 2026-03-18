"""FastAPI application factory for model serving."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Request

from src.serving.registry import ModelAdapter, load_adapter
from src.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)


def _validate_features(
    features: dict[str, float], expected: list[str]
) -> dict[str, float]:
    """Check that all required features are present.

    Extra features are silently ignored.
    Missing features raise a 422 error.
    """
    missing = set(expected) - set(features)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing features: {sorted(missing)}",
        )
    return {k: features[k] for k in expected}


def _predict_single(
    adapter: ModelAdapter, features: dict[str, float]
) -> PredictResponse:
    expected = adapter.feature_names()
    validated = _validate_features(features, expected)
    df = pd.DataFrame([validated])
    proba: np.ndarray = adapter.predict_proba(df)[0]
    predicted_class = int(np.argmax(proba))
    class_names = adapter.class_names()
    return PredictResponse(
        predicted_class=predicted_class,
        predicted_label=class_names[predicted_class],
        probabilities={name: float(p) for name, p in zip(class_names, proba)},
    )


def _get_adapter(request: Request) -> ModelAdapter:
    """Retrieve the model adapter from app state, or raise 503."""
    adapter: ModelAdapter | None = getattr(request.app.state, "adapter", None)
    if adapter is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    return adapter


def _build_router() -> APIRouter:
    """Build the versioned API router."""
    router = APIRouter(prefix="/v1")

    @router.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        adapter = getattr(request.app.state, "adapter", None)
        return HealthResponse(model_loaded=adapter is not None)

    @router.get("/model/info", response_model=ModelInfoResponse)
    async def model_info(request: Request) -> ModelInfoResponse:
        adapter = _get_adapter(request)
        return ModelInfoResponse(
            model_type=request.app.state.model_type,
            features=adapter.feature_names(),
            n_classes=adapter.n_classes(),
            class_names=adapter.class_names(),
        )

    @router.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest, request: Request) -> PredictResponse:
        adapter = _get_adapter(request)
        return _predict_single(adapter, req.features)

    @router.post("/predict/batch", response_model=BatchPredictResponse)
    async def predict_batch(
        req: BatchPredictRequest, request: Request
    ) -> BatchPredictResponse:
        adapter = _get_adapter(request)
        expected = adapter.feature_names()
        validated = [_validate_features(s, expected) for s in req.samples]
        df = pd.DataFrame(validated)
        proba: np.ndarray = adapter.predict_proba(df)
        class_names_list = adapter.class_names()
        predictions = []
        for row in proba:
            predicted_class = int(np.argmax(row))
            predictions.append(
                PredictResponse(
                    predicted_class=predicted_class,
                    predicted_label=class_names_list[predicted_class],
                    probabilities={
                        name: float(p) for name, p in zip(class_names_list, row)
                    },
                )
            )
        return BatchPredictResponse(predictions=predictions)

    return router


def create_app(
    model_type: str = "",
    model_path: str = "",
    class_names: list[str] | None = None,
) -> FastAPI:
    """Create a FastAPI app with lifespan-managed model loading."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        if model_type and model_path:
            app.state.adapter = load_adapter(
                model_type, Path(model_path), class_names=class_names
            )
            app.state.model_type = model_type
        else:
            app.state.adapter = None
            app.state.model_type = ""
        yield
        app.state.adapter = None

    app = FastAPI(
        title="tau-supersymmetry-search inference API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(_build_router())

    return app
