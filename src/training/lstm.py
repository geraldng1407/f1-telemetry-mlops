"""LSTM model for sequential tire-cliff prediction.

Architecture: 2-layer LSTM -> FC head -> scalar output.
Loss: Huber (robust to safety-car outliers).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

DEFAULT_LSTM_PARAMS: dict[str, Any] = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "huber_delta": 4.0,
    "epochs": 80,
    "patience": 10,
    "batch_size": 64,
    "seq_len": 10,
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LSTMCliffModel(nn.Module):
    """2-layer LSTM with a fully-connected regression head."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        _, (h_n, _) = self.lstm(x)
        out = self.head(h_n[-1])  # last layer's final hidden state
        return out.squeeze(-1)


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_lstm(
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    n_features: int,
    params: dict[str, Any] | None = None,
) -> tuple[LSTMCliffModel, TrainHistory]:
    """Train an :class:`LSTMCliffModel` and return the best checkpoint.

    Uses Huber loss, Adam optimiser, ReduceLROnPlateau scheduler, and
    early-stopping on validation loss.
    """
    p = {**DEFAULT_LSTM_PARAMS, **(params or {})}
    device = _get_device()
    logger.info("Training LSTM on {} ({})", device, {k: p[k] for k in ("hidden_size", "num_layers", "lr", "epochs")})

    model = LSTMCliffModel(
        n_features=n_features,
        hidden_size=p["hidden_size"],
        num_layers=p["num_layers"],
        dropout=p["dropout"],
    ).to(device)

    criterion = nn.HuberLoss(delta=p["huber_delta"])
    optimiser = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5,
    )

    history = TrainHistory()
    best_val_loss = float("inf")
    best_state: dict[str, Any] = {}
    epochs_no_improve = 0

    try:
        from tqdm.auto import tqdm
    except ImportError:
        tqdm = None  # type: ignore[assignment]

    epoch_iter = range(p["epochs"])
    if tqdm is not None:
        epoch_iter = tqdm(epoch_iter, desc="LSTM", unit="epoch")

    for epoch in epoch_iter:
        # --- train ---
        model.train()
        train_losses: list[float] = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimiser.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_losses.append(loss.item())

        epoch_train_loss = float(np.mean(train_losses))
        history.train_loss.append(epoch_train_loss)

        # --- validate ---
        epoch_val_loss = float("inf")
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    preds = model(X_batch)
                    val_losses.append(criterion(preds, y_batch).item())
            epoch_val_loss = float(np.mean(val_losses))
        history.val_loss.append(epoch_val_loss)

        scheduler.step(epoch_val_loss if epoch_val_loss < float("inf") else epoch_train_loss)

        if tqdm is not None and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(train=f"{epoch_train_loss:.4f}", val=f"{epoch_val_loss:.4f}")

        # --- early stopping ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history.best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= p["patience"]:
                logger.info("Early stopping at epoch {} (best epoch {})", epoch, history.best_epoch)
                break

    if best_state:
        model.load_state_dict(best_state)
    model = model.to(device)
    model.eval()
    logger.info(
        "LSTM training complete — best val loss {:.4f} at epoch {}",
        best_val_loss,
        history.best_epoch,
    )
    return model, history


# ---------------------------------------------------------------------------
# Predictor wrapper (numpy in / numpy out)
# ---------------------------------------------------------------------------


class LSTMPredictor:
    """Wraps a trained :class:`LSTMCliffModel` with a sklearn-style ``.predict()``."""

    def __init__(self, model: LSTMCliffModel, device: torch.device | None = None) -> None:
        self.model = model
        self.device = device or _get_device()
        self.model.to(self.device).eval()

    def predict(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Run inference on a 3-D numpy array ``(n, seq_len, features)``."""
        self.model.eval()
        preds: list[np.ndarray] = []
        n = len(X)
        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch = torch.as_tensor(
                    X[start : start + batch_size], dtype=torch.float32,
                ).to(self.device)
                out = self.model(batch).cpu().numpy()
                preds.append(out)
        return np.concatenate(preds, axis=0) if preds else np.empty(0, dtype=np.float32)
