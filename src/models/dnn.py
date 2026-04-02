from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import MinMaxScaler

# AMP: use modern API (PyTorch 2.3+) with fallback to legacy API
try:
    from torch.amp import GradScaler, autocast  # PyTorch >= 2.3
except ImportError:
    from torch.cuda.amp import GradScaler, autocast  # PyTorch < 2.3
from tqdm.auto import tqdm

log = logging.getLogger(__name__)


class DNNClassifier(nn.Module):
    """Configurable fully-connected classifier producing raw logits."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_sizes: list[int],
        activation: str = "ReLU",
        dropout: float = 0.0,
    ) -> None:
        """Build the backbone and classification head from the given architecture config."""
        super().__init__()

        self.config: dict = {
            "n_features": n_features,
            "n_classes": n_classes,
            "hidden_sizes": hidden_sizes,
            "activation": activation,
            "dropout": dropout,
        }

        act_cls = getattr(nn, activation)
        layers: list[nn.Module] = []
        in_dim = n_features

        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through the backbone and classification head."""
        return self.head(self.backbone(x))


def resolve_device(cfg: DictConfig | None = None) -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(
    cfg: DictConfig,
    n_features: int,
    n_classes: int,
) -> DNNClassifier:
    """Construct a DNNClassifier from the Hydra model config."""
    mcfg = OmegaConf.to_container(cfg.model, resolve=True)
    return DNNClassifier(
        n_features=n_features,
        n_classes=n_classes,
        hidden_sizes=mcfg["hidden_sizes"],
        activation=mcfg.get("activation", "ReLU"),
        dropout=mcfg.get("dropout", 0.0),
    )


def build_scaler(X_train: pd.DataFrame) -> MinMaxScaler:
    """Fit a MinMaxScaler on training features."""
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    return scaler


def build_criterion(
    class_weights: np.ndarray | None = None,
    device: torch.device = torch.device("cpu"),
) -> nn.CrossEntropyLoss:
    """Build CrossEntropyLoss with optional per-class weights on the given device."""
    weight = None
    if class_weights is not None:
        weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight)


def _batch_indices(
    n: int, batch_size: int, shuffle: bool, generator: torch.Generator | None = None
):
    """Yield index slices for mini-batching a GPU tensor."""
    if shuffle:
        idx = torch.randperm(n, generator=generator)
    else:
        idx = torch.arange(n)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]


def train(
    model: DNNClassifier,
    criterion: nn.CrossEntropyLoss,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scaler: MinMaxScaler,
    cfg: DictConfig,
    device: torch.device,
    early_stopping_rounds: int = 50,
    verbose: bool = True,
) -> tuple[DNNClassifier, dict]:
    """Train a DNN with mini-batch SGD, AMP, and early stopping."""
    mcfg = OmegaConf.to_container(cfg.model, resolve=True)
    n_epochs: int = mcfg["n_epochs"]
    batch_size: int = mcfg["batch_size"]
    use_amp: bool = mcfg.get("amp", False) and device.type == "cuda"

    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    # Normalise and move entire dataset to GPU once
    X_tr = torch.tensor(
        scaler.transform(X_train.values), dtype=torch.float32, device=device
    )
    y_tr = torch.tensor(y_train.to_numpy(), dtype=torch.long, device=device)
    X_va = torch.tensor(
        scaler.transform(X_val.values), dtype=torch.float32, device=device
    )
    y_va = torch.tensor(y_val.to_numpy(), dtype=torch.long, device=device)
    n_train = len(X_tr)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=mcfg["learning_rate"],
        weight_decay=mcfg.get("weight_decay", 0.0),
    )
    grad_scaler = GradScaler(enabled=use_amp)

    # LR scheduler: ReduceLROnPlateau on validation loss
    sched_cfg = mcfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=sched_cfg.get("factor", 0.5),
        patience=sched_cfg.get("patience", 10),
    )

    train_loss_history: list[float] = []
    val_loss_history: list[float] = []
    lr_history: list[float] = []
    best_val_loss = float("inf")
    best_state: dict | None = None
    best_epoch = 0
    patience_counter = 0

    pbar = tqdm(total=n_epochs, desc="Training", unit="epoch") if verbose else None

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for idx in _batch_indices(n_train, batch_size, shuffle=True, generator=gen):
            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(X_tr[idx])
                loss = criterion(logits, y_tr[idx])
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            running_loss += loss.item() * len(idx)

        epoch_train_loss = running_loss / n_train
        train_loss_history.append(epoch_train_loss)

        # --- validation (single pass if it fits, else batched) ---
        model.eval()
        with torch.no_grad(), autocast(device_type=device.type, enabled=use_amp):
            val_logits = model(X_va)
            epoch_val_loss = criterion(val_logits, y_va).item()

        val_loss_history.append(epoch_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(current_lr)
        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if pbar is not None:
            pbar.set_postfix(
                train_loss=f"{epoch_train_loss:.4f}",
                val_loss=f"{epoch_val_loss:.4f}",
                lr=f"{current_lr:.1e}",
            )
            pbar.update(1)

        if patience_counter >= early_stopping_rounds:
            log.info("Early stopping at epoch %d (best epoch: %d)", epoch, best_epoch)
            break

    if pbar is not None:
        pbar.close()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    log.info(
        "Training complete — best epoch: %d, best val loss: %.6f",
        best_epoch,
        best_val_loss,
    )
    history = {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "lr": lr_history,
        "best_epoch": best_epoch,
    }
    return model, history


def train_kfold(
    cfg: DictConfig,
    folds: list[
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]
    ],
    n_classes: int,
    device: torch.device,
    early_stopping_rounds: int = 50,
    verbose: bool = True,
) -> tuple[
    list[DNNClassifier],
    list[MinMaxScaler],
    np.ndarray,
    np.ndarray,
    pd.Series,
    list[dict],
]:
    """Train one DNN per fold and return out-of-fold predictions."""
    models: list[DNNClassifier] = []
    scalers: list[MinMaxScaler] = []
    y_preds: list[np.ndarray] = []
    y_probas: list[np.ndarray] = []
    y_tests: list[pd.Series] = []
    fold_histories: list[dict] = []

    mcfg = OmegaConf.to_container(cfg.model, resolve=True)
    batch_size: int = mcfg["batch_size"]

    fold_pbar = tqdm(total=len(folds), desc="Folds", unit="fold") if verbose else None

    for fold_idx, (X_tr, X_te, y_tr, y_te, w_tr, _) in enumerate(folds):
        log.info("Training fold %d / %d", fold_idx + 1, len(folds))

        n_features = X_tr.shape[1]
        fold_model = build_model(cfg, n_features, n_classes).to(device)
        fold_scaler = build_scaler(X_tr)

        class_weights = w_tr.groupby(y_tr).first().sort_index().values
        fold_criterion = build_criterion(class_weights, device)

        fold_model, history = train(
            fold_model,
            fold_criterion,
            X_tr,
            y_tr,
            X_te,
            y_te,
            scaler=fold_scaler,
            cfg=cfg,
            device=device,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )

        y_pred_fold, y_proba_fold = predict(
            fold_model, X_te, fold_scaler, device, batch_size=batch_size
        )

        models.append(fold_model)
        scalers.append(fold_scaler)
        y_preds.append(y_pred_fold)
        y_probas.append(y_proba_fold)
        y_tests.append(y_te)
        fold_histories.append(history)

        if fold_pbar is not None:
            fold_pbar.update(1)

    if fold_pbar is not None:
        fold_pbar.close()

    y_pred_oof = np.concatenate(y_preds)
    y_proba_oof = np.vstack(y_probas)
    y_test_oof = pd.concat(y_tests)

    return models, scalers, y_pred_oof, y_proba_oof, y_test_oof, fold_histories


def predict(
    model: DNNClassifier,
    X: pd.DataFrame,
    scaler: MinMaxScaler,
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Return hard predictions and probability scores for the given features."""
    model.eval()
    X_t = torch.tensor(scaler.transform(X.values), dtype=torch.float32, device=device)

    probas: list[np.ndarray] = []
    with torch.no_grad():
        for idx in _batch_indices(len(X_t), batch_size, shuffle=False):
            logits = model(X_t[idx])
            probas.append(torch.softmax(logits, dim=1).cpu().numpy())

    y_proba = np.vstack(probas)
    y_pred = np.argmax(y_proba, axis=1)
    return y_pred, y_proba


def save_model(model: DNNClassifier, scaler: MinMaxScaler, path: Path) -> None:
    """Save model weights, config, and fitted scaler to a checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model.config,
            "state_dict": model.state_dict(),
        },
        str(path),
    )
    scaler_path = path.with_suffix(".scaler")
    joblib.dump(scaler, scaler_path)
    log.info("Model saved to %s, scaler to %s", path, scaler_path)


def load_model(
    path: Path,
    device: torch.device = torch.device("cpu"),
) -> tuple[DNNClassifier, MinMaxScaler]:
    """Load model and scaler from a checkpoint and companion scaler file."""
    path = Path(path)
    checkpoint = torch.load(str(path), map_location=device, weights_only=True)
    model = DNNClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    scaler_path = path.with_suffix(".scaler")
    scaler: MinMaxScaler = joblib.load(scaler_path)
    log.info("Model loaded from %s, scaler from %s", path, scaler_path)
    return model, scaler
