"""
Training loop for SpeedNet on IO-VNBD dataset.

- Adam optimizer with cosine annealing LR schedule
- Early stopping on validation RMSE
- Saves best checkpoint to models/speednet_best.pt
- Reports training curve and final test RMSE

Addresses Requirements: SPEED-01, SPEED-03
"""

import os
import json
import time
from typing import Dict, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model import SpeedNet


class SpeedModelTrainer:
    """
    Full training, validation, and evaluation manager for SpeedNet.
    """

    def __init__(
        self,
        model: Optional[SpeedNet] = None,
        window_size: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 60,
        patience: int = 10,
        device: Optional[str] = None,
        checkpoint_dir: str = "models",
    ):
        self.window_size = window_size
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if model is None:
            self.model = SpeedNet(window_size=window_size)
        else:
            self.model = model
        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs, eta_min=1e-5
        )

        self.history: Dict = {"train_rmse": [], "val_rmse": [], "lr": []}
        self.best_val_rmse = float("inf")
        self.best_epoch = 0

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def _run_epoch(self, loader: DataLoader, train: bool = True) -> float:
        """Run one epoch; return RMSE in m/s."""
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        n_samples = 0

        ctx = torch.no_grad() if not train else torch.enable_grad()
        with ctx:
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device, dtype=torch.float32)
                y_batch = y_batch.to(self.device, dtype=torch.float32)

                if train:
                    self.optimizer.zero_grad()

                preds = self.model(X_batch)       # (B, 1)
                loss = self.criterion(preds, y_batch)

                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                batch_n = X_batch.size(0)
                total_loss += loss.item() * batch_n
                n_samples += batch_n

        mse = total_loss / max(n_samples, 1)
        return float(np.sqrt(mse))

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> Dict:
        """
        Train SpeedNet until early stopping or max_epochs.

        Returns:
            history dict with train_rmse, val_rmse, lr lists.
        """
        best_ckpt_path = os.path.join(self.checkpoint_dir, "speednet_best.pt")
        no_improve = 0

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            train_rmse = self._run_epoch(train_loader, train=True)
            val_rmse = self._run_epoch(val_loader, train=False)
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            self.history["train_rmse"].append(train_rmse)
            self.history["val_rmse"].append(val_rmse)
            self.history["lr"].append(current_lr)

            if val_rmse < self.best_val_rmse:
                self.best_val_rmse = val_rmse
                self.best_epoch = epoch
                no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "val_rmse": val_rmse,
                    "window_size": self.window_size,
                }, best_ckpt_path)
            else:
                no_improve += 1

            if verbose and (epoch % 5 == 0 or epoch == 1):
                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch:3d}/{self.max_epochs}  "
                    f"train_RMSE={train_rmse:.4f} m/s  "
                    f"val_RMSE={val_rmse:.4f} m/s  "
                    f"lr={current_lr:.2e}  ({elapsed:.1f}s)"
                )

            if no_improve >= self.patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch} (best val_RMSE={self.best_val_rmse:.4f})")
                break

        return self.history

    def evaluate(self, test_loader: DataLoader, load_best: bool = True) -> Dict:
        """
        Evaluate on test set. Returns metrics dict.

        Returns:
            {rmse_ms, rmse_kmh, mae_ms, max_err_ms}
        """
        best_ckpt_path = os.path.join(self.checkpoint_dir, "speednet_best.pt")
        if load_best and os.path.exists(best_ckpt_path):
            ckpt = torch.load(best_ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])

        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device, dtype=torch.float32)
                preds = self.model(X_batch).cpu().numpy().squeeze()
                labels = y_batch.numpy().squeeze()
                all_preds.append(preds)
                all_labels.append(labels)

        preds_arr = np.concatenate(all_preds)
        labels_arr = np.concatenate(all_labels)
        errors = preds_arr - labels_arr

        rmse_ms = float(np.sqrt(np.mean(errors ** 2)))
        mae_ms = float(np.mean(np.abs(errors)))
        max_err_ms = float(np.max(np.abs(errors)))

        return {
            "rmse_ms": rmse_ms,
            "rmse_kmh": rmse_ms * 3.6,
            "mae_ms": mae_ms,
            "max_err_ms": max_err_ms,
            "best_val_rmse": self.best_val_rmse,
            "best_epoch": self.best_epoch,
        }

    def save_history(self, path: str = "models/training_history.json") -> None:
        """Persist training history to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
