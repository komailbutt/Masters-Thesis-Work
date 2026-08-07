# Sources/trainer.py
# Training loop for CNN Autoencoder (leakage-free, reproducible)
# Features: Hybrid Physical Loss + FIXED LR (no scheduler)
#
# IMPORTANT:
# - This file DOES NOT log to MLflow. MLflow logging is handled in the notebook (Cell 9).
# - Stage-1 focus: CNN-AE for repr="ln_phase" (or other 2-channel real reps).
# - Normalization stats are computed from TRAIN split only (leakage-free).

import os
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================
# 1. Custom Hybrid Physical Loss (optional)
# ==========================================
class PhysicalImpedanceLoss(nn.Module):
    def __init__(self, log_mag_weight=20.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.log_mag_weight = float(log_mag_weight)

    def forward(self, pred_norm: torch.Tensor, target_physical: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculates Hybrid Loss (designed for repr="ln_phase"):

        1) Convert normalized prediction back to physical ln|Z| and phase using TRAIN stats
        2) Convert (ln|Z|, phase) -> (Re, Im) via polar->cartesian
        3) MSE on Re and Im + weighted MSE on ln|Z|

        Args:
            pred_norm:       Model output (normalized) [B, 2, F]
            target_physical: Ground truth (physical)   [B, 2, F]  (ln|Z|, phase)
            stats:           Dict with TRAIN stats: ch0_mean/std, ch1_mean/std
        """
        # 1) Un-normalize prediction to physical ln|Z| and phase
        pred_ln = pred_norm[:, 0, :] * (stats["ch0_std"] + 1e-6) + stats["ch0_mean"]
        pred_ph = pred_norm[:, 1, :] * (stats["ch1_std"] + 1e-6) + stats["ch1_mean"]

        # 2) Targets are already physical (ln|Z|, phase)
        tgt_ln = target_physical[:, 0, :]
        tgt_ph = target_physical[:, 1, :]

        # 3) Polar -> Cartesian
        pred_mag = torch.exp(pred_ln)
        pred_re = pred_mag * torch.cos(pred_ph)
        pred_im = pred_mag * torch.sin(pred_ph)

        tgt_mag = torch.exp(tgt_ln)
        tgt_re = tgt_mag * torch.cos(tgt_ph)
        tgt_im = tgt_mag * torch.sin(tgt_ph)

        # 4) Loss parts
        loss_re = self.mse(pred_re, tgt_re)
        loss_im = self.mse(pred_im, tgt_im)
        loss_ln = self.mse(pred_ln, tgt_ln)

        return loss_re + loss_im + self.log_mag_weight * loss_ln


# ==========================================
# 2. Loss Builder
# ==========================================
def _build_loss(config: dict) -> Tuple[nn.Module, str]:
    """
    Returns:
        criterion : torch loss module
        loss_tag  : string used for printing/saving
    """
    loss_name = str(config.get("loss_name", "mse")).lower()

    # Option 1: MSE
    if loss_name in ["mse", "mse_loss"]:
        return nn.MSELoss(), "mse"

    # Option 2: Huber
    if loss_name in ["huber", "huber_loss"]:
        delta = float(config.get("huber_delta", 1.0))
        return nn.HuberLoss(delta=delta), f"huber(delta={delta})"

    # Option 3: Physical/Hybrid (ln_phase only)
    if any(x in loss_name for x in ["physical", "phy", "hybrid", "physical_hybrid"]):
        w = float(config.get("log_mag_weight", 20.0))
        return PhysicalImpedanceLoss(log_mag_weight=w), f"physical_hybrid_w{w:g}"

    raise ValueError(f"Unknown loss_name='{loss_name}'. Use 'mse', 'huber', or 'physical'.")


# ==========================================
# 3. Helpers: train-only stats + normalization
# ==========================================
@torch.no_grad()
def _compute_train_stats(train_loader, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Compute TRAIN-only normalization stats (leakage-free).
    x is real-valued with shape [B, 2, F].

    Returns:
        ch0_mean, ch0_std, ch1_mean, ch1_std   (on device)
    """
    ch0_list, ch1_list = [], []
    for x, _, _ in train_loader:
        ch0_list.append(x[:, 0, :].cpu())
        ch1_list.append(x[:, 1, :].cpu())

    ch0 = torch.cat(ch0_list, dim=0)
    ch1 = torch.cat(ch1_list, dim=0)

    ch0_mean, ch0_std = ch0.mean(), ch0.std()
    ch1_mean, ch1_std = ch1.mean(), ch1.std()

    # avoid div0
    if float(ch0_std) == 0.0:
        ch0_std = torch.tensor(1.0)
    if float(ch1_std) == 0.0:
        ch1_std = torch.tensor(1.0)

    return {
        "ch0_mean": ch0_mean.to(device),
        "ch0_std":  ch0_std.to(device),
        "ch1_mean": ch1_mean.to(device),
        "ch1_std":  ch1_std.to(device),
    }


def _normalize_batch(x: torch.Tensor, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Normalize a batch using TRAIN stats, without modifying stored tensors.
    """
    x_norm = x.clone()
    x_norm[:, 0, :] = (x_norm[:, 0, :] - stats["ch0_mean"]) / (stats["ch0_std"] + 1e-6)
    x_norm[:, 1, :] = (x_norm[:, 1, :] - stats["ch1_mean"]) / (stats["ch1_std"] + 1e-6)
    return x_norm


# ==========================================
# 4. Train Function
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
    Trains CNN Autoencoder with leakage-free normalization and FIXED LR (no scheduler).

    Returns:
        history : dict (train_loss, val_loss, lr)
        stats   : dict of TRAIN normalization stats (on device)
        times   : dict with timing + best_val_loss + best_epoch
    """
    criterion, loss_tag = _build_loss(config)

    lr = float(config["lr"])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epochs = int(config["epochs"])
    is_physical_loss = isinstance(criterion, PhysicalImpedanceLoss)

    patience = int(config.get("patience", 30))
    min_delta = float(config.get("min_delta", 1e-6))
    log_every = int(config.get("log_every", 10))

    # Model Naming
    model_tag = "FULL" if subset_size is None else f"N{subset_size}"
    if run_name is None:
        run_name = f"cnn_ae_{model_tag}"
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

    print(f"Training CNN-AE on dataset: {model_tag}")
    print(f"Checkpoint file: {model_filename}")
    print(f"Loss: {loss_tag}")
    print(f"Learning Rate: FIXED {lr:.2e} (no scheduler)")
    print(f"Early stopping: patience={patience}, min_delta={min_delta:e}")
    print(f"Logging: print every {log_every} epochs")

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

            if is_physical_loss:
                # compares recon_norm against physical x (ln|Z|, phase)
                loss = criterion(recon_norm, x, stats)
            else:
                # compares in normalized domain
                loss = criterion(recon_norm, x_norm)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)   #added for LSTM
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

                if is_physical_loss:
                    loss = criterion(recon_norm, x, stats)
                else:
                    loss = criterion(recon_norm, x_norm)

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
