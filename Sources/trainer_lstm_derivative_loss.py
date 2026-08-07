# Sources/trainer_lstm_1.py
# Training loop for LSTM Autoencoder (leakage-free, reproducible)
# Features: FIXED LR (no scheduler), gradient clipping (useful for RNNs)
#
# IMPORTANT:
# - This file DOES NOT log to MLflow. MLflow logging is handled in the notebook.
# - This trainer supports C=2 or C=3 channels, but is meant for LSTM with repr="ln_sincos".
# - Normalization stats are computed from TRAIN split only (leakage-free).
# - NEW: supports weighted loss along frequency axis so high-frequency region can be emphasized.
# - NEW: supports derivative loss so sharp local spike / transition features are not over-smoothed.

import os
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================
# 1. Frequency-Weighted MSE Loss (NEW)
# ==========================================
class FrequencyWeightedMSELoss(nn.Module):
    """
    Weighted MSE over frequency axis.

    Expected tensor shape:
        pred, target : [B, C, F]

    Weight vector shape:
        freq_weights : [F]
    """
    def __init__(self, freq_weights: torch.Tensor):
        super().__init__()

        if freq_weights.ndim != 1:
            raise ValueError(f"freq_weights must be 1D [F], got shape {tuple(freq_weights.shape)}")

        # store as buffer so it moves with model/device safely
        self.register_buffer("freq_weights", freq_weights.float())

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")

        if pred.ndim != 3:
            raise ValueError(f"Expected pred/target shape [B, C, F], got {tuple(pred.shape)}")

        if pred.shape[-1] != self.freq_weights.shape[0]:
            raise ValueError(
                f"Frequency dim mismatch: pred.shape[-1]={pred.shape[-1]} "
                f"vs len(freq_weights)={self.freq_weights.shape[0]}"
            )

        # [F] -> [1,1,F] so it broadcasts across batch and channels
        w = self.freq_weights.view(1, 1, -1)

        loss = w * (pred - target) ** 2
        return loss.mean()


# ==========================================
# 2. Helper: build frequency weights (NEW)
# ==========================================
def _build_frequency_weights(config: dict, freq_len: int, device: torch.device) -> torch.Tensor:
    """
    Build simple piecewise weights over frequency axis.

    Default behavior:
        first 10%   -> low band
        next 20%    -> mid band
        last 70%    -> high band

    You can tune this later if needed.
    """
    low_w  = float(config.get("low_band_weight", 1.0))
    mid_w  = float(config.get("mid_band_weight", 1.0))
    high_w = float(config.get("high_band_weight", 2.0))

    w = torch.ones(freq_len, dtype=torch.float32, device=device)

    i1 = int(0.10 * freq_len)   # ~ first 10%
    i2 = int(0.30 * freq_len)   # ~ next 20%
    # remaining ~70% = high band

    w[:i1] = low_w
    w[i1:i2] = mid_w
    w[i2:] = high_w

    return w


# ==========================================
# 3. Loss Builder (same as CNN trainer + NEW weighted_mse)
# ==========================================
def _build_loss(config: dict, freq_len: int = None, device: torch.device = None) -> Tuple[nn.Module, str]:
    """
    Returns:
        criterion : torch loss module
        loss_tag  : string used for printing/saving
    """
    loss_name = str(config.get("loss_name", "mse")).lower()

    if loss_name in ["mse", "mse_loss"]:
        return nn.MSELoss(), "mse"

    if loss_name in ["huber", "huber_loss"]:
        delta = float(config.get("huber_delta", 1.0))
        return nn.HuberLoss(delta=delta), f"huber(delta={delta})"

    if loss_name in ["weighted_mse", "freq_weighted_mse", "weighted_freq_mse"]:
        if freq_len is None or device is None:
            raise ValueError("freq_len and device are required for weighted_mse")

        freq_weights = _build_frequency_weights(config, freq_len=freq_len, device=device)

        low_w  = float(config.get("low_band_weight", 1.0))
        mid_w  = float(config.get("mid_band_weight", 1.0))
        high_w = float(config.get("high_band_weight", 2.0))

        return (
            FrequencyWeightedMSELoss(freq_weights=freq_weights),
            f"weighted_mse(low={low_w},mid={mid_w},high={high_w})"
        )

    raise ValueError(f"Unknown loss_name='{loss_name}'. Use 'mse' or 'huber' or 'weighted_mse'.")


# ==========================================
# 4. Helpers: derivative loss (NEW)
# ==========================================
def _first_difference(x: torch.Tensor) -> torch.Tensor:
    """
    First difference along frequency axis.

    Args:
        x : [B, C, F]

    Returns:
        dx : [B, C, F-1]
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x shape [B, C, F], got {tuple(x.shape)}")
    return x[:, :, 1:] - x[:, :, :-1]


def _derivative_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    use_freq_weights: bool = False,
    freq_weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    Derivative matching loss along frequency axis.

    Args:
        pred, target : [B, C, F]
        use_freq_weights : whether to apply frequency weights to derivative domain
        freq_weights     : [F]

    Returns:
        scalar loss
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}")

    pred_dx = _first_difference(pred)     # [B, C, F-1]
    targ_dx = _first_difference(target)   # [B, C, F-1]

    diff_sq = (pred_dx - targ_dx) ** 2

    if use_freq_weights:
        if freq_weights is None:
            raise ValueError("freq_weights must be provided when use_freq_weights=True")
        if freq_weights.ndim != 1:
            raise ValueError(f"freq_weights must be 1D [F], got {tuple(freq_weights.shape)}")

        # derivative lives on intervals -> use weights for F-1 positions
        # simplest consistent choice: drop the first weight
        w_diff = freq_weights[1:]                  # [F-1]
        w_diff = w_diff.view(1, 1, -1)            # [1,1,F-1]
        diff_sq = diff_sq * w_diff

    return diff_sq.mean()


# ==========================================
# 5. Helpers: train-only stats + normalization
# ==========================================
@torch.no_grad()
def _compute_train_stats(train_loader, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Compute TRAIN-only normalization stats (leakage-free).
    x is real-valued with shape [B, C, F]. (C can be 2 or 3)

    Returns:
        stats dict: ch0_mean/std, ch1_mean/std, ... (on device)
    """
    # Peek one batch to infer channel count
    x0, _, _ = next(iter(train_loader))
    C = int(x0.shape[1])

    ch_list = [[] for _ in range(C)]
    for x, _, _ in train_loader:
        for c in range(C):
            ch_list[c].append(x[:, c, :].cpu())

    stats: Dict[str, torch.Tensor] = {}
    for c in range(C):
        ch = torch.cat(ch_list[c], dim=0)
        mean, std = ch.mean(), ch.std()

        if float(std) == 0.0:
            std = torch.tensor(1.0)

        stats[f"ch{c}_mean"] = mean.to(device)
        stats[f"ch{c}_std"]  = std.to(device)

    return stats


def _normalize_batch(x: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Normalize a batch using TRAIN stats, without modifying stored tensors.
    x: [B, C, F]
    """
    x_norm = x.clone()
    C = int(x_norm.shape[1])
    for c in range(C):
        x_norm[:, c, :] = (x_norm[:, c, :] - stats[f"ch{c}_mean"]) / (stats[f"ch{c}_std"] + 1e-6)
    return x_norm


# ==========================================
# 6. Train Function
# ==========================================
def train_model(
    model,
    train_loader,
    val_loader,
    config,
    device,
    subset_size=None,
    run_name=None,
    checkpoint_dir=None,
    run_cfg=None,
):
    """
    Trains LSTM Autoencoder with leakage-free normalization and FIXED LR (no scheduler).

    Returns:
        history : dict (train_loss, val_loss, lr)
        stats   : dict of TRAIN normalization stats (on device)
        times   : dict with timing + best_val_loss + best_epoch
    """
    # infer frequency length from one batch
    x0, _, _ = next(iter(train_loader))
    freq_len = int(x0.shape[-1])

    criterion, loss_tag = _build_loss(config, freq_len=freq_len, device=device)

    lr = float(config["lr"])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = int(config["epochs"])
    patience = int(config.get("patience", 30))
    min_delta = float(config.get("min_delta", 1e-6))
    log_every = int(config.get("log_every", 10))

    # RNN-specific: grad clipping helps stability
    grad_clip = float(config.get("grad_clip", 1.0))

    # NEW: derivative-loss controls
    use_derivative_loss = bool(config.get("use_derivative_loss", False))
    lambda_diff = float(config.get("lambda_diff", 0.0))

    # Model Naming
    model_tag = "FULL" if subset_size is None else f"N{subset_size}"
    if run_name is None:
        run_name = f"lstm_ae_{model_tag}"
    model_filename = f"{run_name}.pt"

    if checkpoint_dir is None:
        checkpoint_dir = os.path.abspath(os.path.join(os.getcwd(), "checkpoints"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, model_filename)

    # -------- Compute TRAIN-only normalization stats --------
    model.eval()
    stats = _compute_train_stats(train_loader, device=device)

    # -------- Training Loop --------
    history = {"train_loss": [], "val_loss": [], "lr": []}
    best_val_loss = float("inf")
    best_epoch = None
    epochs_no_improve = 0
    stopped_epoch = None

    print(f"Training LSTM-AE on dataset: {model_tag}")
    print(f"Checkpoint file: {model_filename}")
    print(f"Loss: {loss_tag}")
    print(f"Learning Rate: FIXED {lr:.2e} (no scheduler)")
    print(f"Grad clip: {grad_clip:g}")
    print(f"Early stopping: patience={patience}, min_delta={min_delta:e}")
    print(f"Logging: print every {log_every} epochs")

    # NEW: if weighted loss is used, print the actual weights for transparency
    if isinstance(criterion, FrequencyWeightedMSELoss):
        low_w  = float(config.get("low_band_weight", 1.0))
        mid_w  = float(config.get("mid_band_weight", 1.0))
        high_w = float(config.get("high_band_weight", 2.0))
        print(f"Frequency weighting active: low={low_w}, mid={mid_w}, high={high_w}")

    # NEW: derivative loss transparency
    if use_derivative_loss and lambda_diff > 0.0:
        print(f"Derivative loss active: lambda_diff={lambda_diff}")

    start_time = time.time()

    for epoch in range(epochs):
        # ---- TRAIN ----
        model.train()
        train_loss_sum = 0.0

        for x, _, _ in train_loader:
            x = x.to(device)
            x_norm = _normalize_batch(x, stats)

            optimizer.zero_grad()
            recon_norm, _ = model(x_norm)

            # base reconstruction loss
            loss_recon = criterion(recon_norm, x_norm)
            loss = loss_recon

            # NEW: derivative loss on top of base reconstruction loss
            if use_derivative_loss and lambda_diff > 0.0:
                if isinstance(criterion, FrequencyWeightedMSELoss):
                    loss_diff = _derivative_loss(
                        pred=recon_norm,
                        target=x_norm,
                        use_freq_weights=True,
                        freq_weights=criterion.freq_weights,
                    )
                else:
                    loss_diff = _derivative_loss(
                        pred=recon_norm,
                        target=x_norm,
                        use_freq_weights=False,
                        freq_weights=None,
                    )

                loss = loss_recon + lambda_diff * loss_diff

            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            train_loss_sum += float(loss.item())

        avg_train = train_loss_sum / max(len(train_loader), 1)

        # ---- VALIDATION ----
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, _, _ in val_loader:
                x = x.to(device)
                x_norm = _normalize_batch(x, stats)
                recon_norm, _ = model(x_norm)

                # base reconstruction loss
                loss_recon = criterion(recon_norm, x_norm)
                loss = loss_recon

                # NEW: derivative loss on validation too
                if use_derivative_loss and lambda_diff > 0.0:
                    if isinstance(criterion, FrequencyWeightedMSELoss):
                        loss_diff = _derivative_loss(
                            pred=recon_norm,
                            target=x_norm,
                            use_freq_weights=True,
                            freq_weights=criterion.freq_weights,
                        )
                    else:
                        loss_diff = _derivative_loss(
                            pred=recon_norm,
                            target=x_norm,
                            use_freq_weights=False,
                            freq_weights=None,
                        )

                    loss = loss_recon + lambda_diff * loss_diff

                val_loss_sum += float(loss.item())

        avg_val = val_loss_sum / max(len(val_loader), 1)
        current_lr = float(optimizer.param_groups[0]["lr"])

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["lr"].append(current_lr)

        # ---- SAVE BEST + EARLY STOPPING ----
        improved = (best_val_loss - avg_val) > min_delta
        if improved:
            best_val_loss = avg_val
            best_epoch = epoch + 1
            epochs_no_improve = 0

            # Save stats on CPU for portability
            stats_cpu = {k: v.detach().cpu() for k, v in stats.items()}

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "stats": stats_cpu,
                    "config": dict(config),
                    "subset_size": subset_size,
                    "run_name": run_name,
                    "history": history,
                    "times": {"best_epoch": best_epoch, "best_val_loss": float(best_val_loss)},
                    "loss_tag": loss_tag,
                },
                ckpt_path,
            )

            if (epoch + 1) % log_every == 0 or (epoch + 1) == 1:
                print(f"[best updated] ep={epoch+1} | best_val={best_val_loss:.6f} | lr={current_lr:.2e}")
        else:
            epochs_no_improve += 1

        # ---- PRINT ----
        if (epoch + 1) % log_every == 0:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train: {avg_train:.6f} | "
                f"Val: {avg_val:.6f} | "
                f"no_imp={epochs_no_improve}/{patience}"
            )

        if epochs_no_improve >= patience:
            stopped_epoch = epoch + 1
            print(f"Early stopping triggered at epoch {stopped_epoch}. (best_epoch={best_epoch})")
            break

    total_time = time.time() - start_time
    times = {
        "train_time_sec": float(total_time),
        "train_time_min": float(total_time / 60.0),
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "stopped_epoch": stopped_epoch,
    }

    print(f"Training completed in {times['train_time_sec']:.2f} s ({times['train_time_min']:.2f} min)")
    print(f"Best validation loss (normalized-domain): {best_val_loss:.6f} at epoch {best_epoch}")

    return history, stats, times