"""Model adapter pattern for uniform BDT / DNN inference."""

from __future__ import annotations

import abc
from pathlib import Path

import numpy as np
import pandas as pd


class ModelAdapter(abc.ABC):
    """Uniform interface around any trained classifier."""

    @abc.abstractmethod
    def feature_names(self) -> list[str]:
        """Return the ordered list of feature names the model expects."""

    @abc.abstractmethod
    def n_classes(self) -> int:
        """Return the number of output classes."""

    @abc.abstractmethod
    def class_names(self) -> list[str]:
        """Return human-readable class labels."""

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return shape (n_samples, n_classes) probability array."""


class BDTAdapter(ModelAdapter):
    """Adapter for an XGBoost BDT checkpoint (.ubj)."""

    def __init__(self, path: Path, class_names: list[str] | None = None) -> None:
        from src.models.bdt import load_model

        self._model = load_model(path)
        self._class_names = class_names

    def feature_names(self) -> list[str]:
        return list(self._model.get_booster().feature_names)

    def n_classes(self) -> int:
        # XGBoost n_classes_ attribute
        n = getattr(self._model, "n_classes_", None)
        if n is not None:
            return int(n)
        # Fallback: infer from objective
        obj = self._model.get_params().get("objective", "")
        if "multi" in obj:
            return int(self._model.get_params().get("num_class", 2))
        return 2

    def class_names(self) -> list[str]:
        if self._class_names:
            return self._class_names
        return [str(i) for i in range(self.n_classes())]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)


class DNNAdapter(ModelAdapter):
    """Adapter for a PyTorch DNN checkpoint (.pt)."""

    def __init__(self, path: Path, class_names: list[str] | None = None) -> None:
        import torch

        from src.models.dnn import load_model

        self._device = torch.device("cpu")
        self._model, self._scaler = load_model(path, device=self._device)
        self._class_names = class_names

    def feature_names(self) -> list[str]:
        names = getattr(self._scaler, "feature_names_in_", None)
        if names is not None:
            return list(names)
        # Fallback: generic names from input dimension
        n = self._model.config["n_features"]
        return [f"feature_{i}" for i in range(n)]

    def n_classes(self) -> int:
        return self._model.config["n_classes"]

    def class_names(self) -> list[str]:
        if self._class_names:
            return self._class_names
        return [str(i) for i in range(self.n_classes())]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        self._model.eval()
        X_scaled = self._scaler.transform(X)
        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            logits = self._model(X_t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba


ADAPTERS: dict[str, type[ModelAdapter]] = {
    "bdt": BDTAdapter,
    "dnn": DNNAdapter,
}


def load_adapter(
    model_type: str,
    model_path: str | Path,
    class_names: list[str] | None = None,
) -> ModelAdapter:
    """Factory: load a model behind the uniform adapter interface."""
    cls = ADAPTERS.get(model_type)
    if cls is None:
        raise ValueError(
            f"Unknown model type {model_type!r}. Choose from: {list(ADAPTERS)}"
        )
    return cls(Path(model_path), class_names=class_names)
