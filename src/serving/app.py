"""FastAPI application factory for model serving."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.serving.registry import ModelAdapter, load_adapter
from src.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)


@dataclass
class _State:
    adapter: ModelAdapter | None = None
    model_type: str = ""


_state = _State()


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


def _predict_single(features: dict[str, float]) -> PredictResponse:
    adapter = _state.adapter
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


def create_app(
    model_type: str = "",
    model_path: str = "",
    class_names: list[str] | None = None,
) -> FastAPI:
    """Create a FastAPI app with lifespan-managed model loading."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        # Load model on startup
        if model_type and model_path:
            _state.adapter = load_adapter(
                model_type, Path(model_path), class_names=class_names
            )
            _state.model_type = model_type
        yield
        # Cleanup
        _state.adapter = None

    app = FastAPI(
        title="tau-supersymmetry inference API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(model_loaded=_state.adapter is not None)

    @app.get("/model/info", response_model=ModelInfoResponse)
    async def model_info() -> ModelInfoResponse:
        if _state.adapter is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        adapter = _state.adapter
        return ModelInfoResponse(
            model_type=_state.model_type,
            features=adapter.feature_names(),
            n_classes=adapter.n_classes(),
            class_names=adapter.class_names(),
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest) -> PredictResponse:
        if _state.adapter is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        return _predict_single(req.features)

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    async def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
        if _state.adapter is None:
            raise HTTPException(status_code=503, detail="No model loaded")
        adapter = _state.adapter
        expected = adapter.feature_names()
        # Validate all samples
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

    return app
