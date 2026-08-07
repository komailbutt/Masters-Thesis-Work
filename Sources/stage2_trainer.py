# Sources/stage2_trainer.py
# Training loop for Stage-2: geometry/material parameters -> latent vector z

from __future__ import annotations

import os
import time
import copy
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn


class EarlyStoppingState:
    def __init__(self, patience: int = 80, min_delta: float = 1e-6):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.num_bad = 0
        self.best_epoch = None

    def step(self, value: float, epoch: int) -> bool:
        improved = (self.best - value) > self.min_delta

        if improved:
            self.best = float(value)
            self.num_bad = 0
            self.best_epoch = int(epoch)
        else:
            self.num_bad += 1

        return self.num_bad >= self.patience


def _build_optimizer(model, config: Dict[str, Any]):
    opt_name = str(config.get("optimizer", "adam")).lower()
    lr = float(config.get("lr", 5e-4))
    weight_decay = float(config.get("weight_decay", 1e-5))

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    raise ValueError("optimizer must be 'adam' or 'adamw'.")


def _build_loss(config: Dict[str, Any]):
    loss_name = str(config.get("loss_function", "mse")).lower()

    if loss_name in ["mse", "mse_loss"]:
        return nn.MSELoss(), "MSE"

    if loss_name in ["huber", "huber_loss"]:
        delta = float(config.get("huber_delta", 1.0))
        return nn.HuberLoss(delta=delta), f"Huber(delta={delta})"

    raise ValueError("loss_function must be 'mse' or 'huber'.")


def train_stage2_model(
    model: nn.Module,
    train_loader,
    val_loader,
    config: Dict[str, Any],
    device: torch.device,
    checkpoint_dir: str,
    run_name: str,
    mlflow_log: bool = False,
) -> Tuple[nn.Module, Dict[str, list], Dict[str, Any], str]:
    """
    Train Stage-2 model with best checkpoint saving.

    Dataloader batch:
        xb      : normalized geometry/material vector
        zb      : normalized latent vector
        row_idx : original database row index
        sid     : simulation ID
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{run_name}.pt")

    criterion, loss_tag = _build_loss(config)
    optimizer = _build_optimizer(model, config)

    epochs = int(config.get("epochs", 1000))
    patience = int(config.get("patience", 80))
    min_delta = float(config.get("min_delta", 1e-6))
    log_every = int(config.get("log_every", 10))
    grad_clip = float(config.get("grad_clip", 1.0))

    stopper = EarlyStoppingState(patience=patience, min_delta=min_delta)

    model.to(device)

    best_state = None
    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    start = time.time()
    stopped_epoch = None

    print(f"Training Stage-2 model: {run_name}")
    print(f"Loss: {loss_tag}")
    print(f"Checkpoint: {ckpt_path}")

    for epoch in range(1, epochs + 1):
        # ---------------- TRAIN ----------------
        model.train()

        train_sum = 0.0
        n_train = 0

        for xb, zb, _, _ in train_loader:
            xb = xb.to(device)
            zb = zb.to(device)

            optimizer.zero_grad()  #clear previous gradients
            pred = model(xb)
            loss = criterion(pred, zb)
            loss.backward() #new gradients computed here

            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            train_sum += float(loss.item()) * xb.size(0)
            n_train += xb.size(0)

        train_loss = train_sum / max(n_train, 1)

        # ---------------- VAL ----------------
        model.eval()

        val_sum = 0.0
        n_val = 0

        with torch.no_grad():
            for xb, zb, _, _ in val_loader:
                xb = xb.to(device)
                zb = zb.to(device)

                pred = model(xb)
                loss = criterion(pred, zb)

                val_sum += float(loss.item()) * xb.size(0)
                n_val += xb.size(0)

        val_loss = val_sum / max(n_val, 1)
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # ---------------- BEST CHECKPOINT ----------------
        if val_loss < stopper.best - stopper.min_delta:
            best_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "model_state": best_state,
                    "config": dict(config),
                    "run_name": run_name,
                    "epoch": int(epoch),
                    "best_val_loss": float(val_loss),
                    "history": history,
                },
                ckpt_path,
            )

        should_stop = stopper.step(val_loss, epoch)

        if mlflow_log:
            try:
                import mlflow
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("lr", current_lr, step=epoch)
            except Exception:
                pass

        if epoch == 1 or epoch % log_every == 0:
            print(
                f"Epoch {epoch:04d}/{epochs} | "
                f"train={train_loss:.6f} | "
                f"val={val_loss:.6f} | "
                f"bad={stopper.num_bad}/{patience}"
            )

        if should_stop:
            stopped_epoch = epoch
            print(f"Early stopping at epoch {epoch}. Best epoch: {stopper.best_epoch}")
            break

    elapsed = time.time() - start

    if best_state is not None:
        model.load_state_dict(best_state)

    actual_stopped_epoch = stopped_epoch if stopped_epoch is not None else len(history["val_loss"])

    times = {
        "train_time_sec": float(elapsed),
        "train_time_min": float(elapsed / 60.0),
        "best_epoch": int(stopper.best_epoch),
        "best_val_loss": float(stopper.best),
        "stopped_epoch": int(actual_stopped_epoch),
        "early_stopped": int(stopped_epoch is not None),
    }

    # enrich checkpoint with final timing
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt["times"] = times
        torch.save(ckpt, ckpt_path)

    print(
        f"Done. Best val={stopper.best:.6f} "
        f"at epoch {stopper.best_epoch}. "
        f"Stopped epoch={actual_stopped_epoch}. "
        f"Time={elapsed / 60:.2f} min"
    )

    return model, history, times, ckpt_path


@torch.no_grad()
def predict_latents(
    model: nn.Module,
    loader,
    device: torch.device,
    z_scaler=None,
) -> Dict[str, np.ndarray]:
    """
    Predict latent vectors on a loader.
    Returns both normalized and inverse-scaled latent predictions.
    """
    model.eval()

    pred_norm = []
    true_norm = []
    row_idx = []
    sid = []

    start = time.time()

    for xb, zb, idx, sids in loader:
        xb = xb.to(device)

        pred = model(xb).cpu().numpy()

        pred_norm.append(pred)
        true_norm.append(zb.numpy())
        row_idx.append(idx.numpy())
        sid.append(sids.numpy())

    prediction_time_sec = time.time() - start

    pred_norm = np.concatenate(pred_norm, axis=0)
    true_norm = np.concatenate(true_norm, axis=0)
    row_idx = np.concatenate(row_idx, axis=0)
    sid = np.concatenate(sid, axis=0)

    if z_scaler is not None:
        pred = z_scaler.inverse_transform(pred_norm)
        true = z_scaler.inverse_transform(true_norm)
    else:
        pred = pred_norm
        true = true_norm

    return {
        "z_pred_norm": pred_norm.astype(np.float32),
        "z_true_norm": true_norm.astype(np.float32),
        "z_pred": pred.astype(np.float32),
        "z_true": true.astype(np.float32),
        "row_idx": row_idx.astype(np.int64),
        "sid": sid.astype(np.int64),
        "latent_prediction_time_sec": float(prediction_time_sec),
        "latent_prediction_time_per_sample_sec": float(prediction_time_sec / max(len(sid), 1)),
    }


def compute_latent_metrics(z_true: np.ndarray, z_pred: np.ndarray) -> Dict[str, float]:
    err = z_pred - z_true
    abs_err = np.abs(err)
    l2 = np.linalg.norm(err, axis=1)

    return {
        "latent_mae": float(abs_err.mean()),
        "latent_mse": float(np.mean(err ** 2)),
        "latent_rmse": float(np.sqrt(np.mean(err ** 2))),
        "latent_mean_l2": float(l2.mean()),
        "latent_median_l2": float(np.median(l2)),
        "latent_p95_l2": float(np.percentile(l2, 95)),
    }


def save_stage2_checkpoint_package(
    checkpoint_path: str,
    model: nn.Module,
    config: Dict[str, Any],
    run_name: str,
    feature_cols,
    x_scaler,
    z_scaler,
    split_info,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Store model + scalers + split metadata in one .pt checkpoint.
    No separate joblib scalers are needed.
    """
    pkg = {
        "model_state": model.state_dict(),
        "config": dict(config),
        "run_name": run_name,
        "feature_cols": list(feature_cols),
        "split_info": dict(split_info),

        "x_scaler_mean": x_scaler.mean_,
        "x_scaler_scale": x_scaler.scale_,
        "z_scaler_mean": z_scaler.mean_,
        "z_scaler_scale": z_scaler.scale_,
    }

    if extra:
        pkg.update(extra)

    torch.save(pkg, checkpoint_path)