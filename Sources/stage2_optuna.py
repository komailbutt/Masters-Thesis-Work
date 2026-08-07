# Sources/stage2_optuna.py
#
# Optuna utilities for Stage-2 hyperparameter tuning:
#   geometry/material parameters -> latent vector -> CNN-AE decoder -> impedance
#
# This file is intentionally separate from stage2_dataset.py, stage2_models.py,
# and stage2_trainer.py so your current baseline notebook is not disturbed.
#
# Optuna objective:
#   minimize validation decoded score:
#       score = MAE_dB + phase_weight * MAE_phase_deg
#
# Why decoded validation score?
# - The final application is impedance prediction.
# - Latent loss alone may not reflect decoded impedance quality.
# - Test set remains untouched.

from __future__ import annotations

import os
import time
import copy
import random
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import optuna
import mlflow

from Sources.stage2_models import build_stage2_model, count_trainable_parameters
from Sources.metrics import compute_metrics


def set_global_seed(seed: int = 42) -> None:
    """Make trial training more reproducible."""
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def suggest_stage2_hparams(trial: optuna.Trial, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Defines Optuna search space.

    Notes:
    - We tune architecture as categorical templates, not arbitrary layers,
      because your current good models are structured and deep.
    - We keep the search space compact enough for 20k samples and GPU training.
    """

    arch_name = trial.suggest_categorical(
        "architecture",
        [
            "A_256_512_512_512_512_256_256",
            "B_512_512_512_512_512_256_256",
            "C_256_512_1024_512_256",
            "D_128_512_512_512_256",
            "E_64_512_768_768_512_256",
            "F_256_512_768_768_512_256",
        ],
    )

    arch_map = {
        "A_256_512_512_512_512_256_256": (256, 512, 512, 512, 512, 256, 256),
        "B_512_512_512_512_512_256_256": (512, 512, 512, 512, 512, 256, 256),
        "C_256_512_1024_512_256": (256, 512, 1024, 512, 256),
        "D_128_512_512_512_256": (128, 512, 512, 512, 256),
        "E_64_512_768_768_512_256": (64, 512, 768, 768, 512, 256),
        "F_256_512_768_768_512_256": (256, 512, 768, 768, 512, 256),
    }

    loss_function = trial.suggest_categorical("loss_function", ["huber", "mse"])

    trial_cfg = dict(base_cfg)

    trial_cfg.update(
        {
            "hidden_dims": arch_map[arch_name],
            "architecture": arch_name,
            "activation": trial.suggest_categorical("activation", ["relu", "gelu", "silu"]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.06, step=0.01),
            "use_batchnorm": trial.suggest_categorical("use_batchnorm", [False, True]),

            "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
            "loss_function": loss_function,
            "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 3e-4, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
            "grad_clip": trial.suggest_categorical("grad_clip", [0.5, 1.0, 2.0]),

            # Shorter than final training. Best config should be retrained later.
            "epochs": int(base_cfg.get("optuna_trial_epochs", 500)),
            "patience": int(base_cfg.get("optuna_patience", 50)),
            "min_delta": float(base_cfg.get("min_delta", 1e-6)),
            "log_every": int(base_cfg.get("optuna_log_every", 100)),
        }
    )

    if loss_function == "huber":
        trial_cfg["huber_delta"] = trial.suggest_categorical("huber_delta", [0.25, 0.5, 1.0, 2.0])
    else:
        trial_cfg["huber_delta"] = None

    return trial_cfg


def _build_loss(config: Dict[str, Any]) -> Tuple[nn.Module, str]:
    loss_name = str(config.get("loss_function", "mse")).lower()

    if loss_name == "mse":
        return nn.MSELoss(), "mse"

    if loss_name == "huber":
        delta = float(config.get("huber_delta", 1.0))
        return nn.HuberLoss(delta=delta), f"huber(delta={delta})"

    raise ValueError("loss_function must be 'mse' or 'huber'.")


def _build_optimizer(model: nn.Module, config: Dict[str, Any]):
    opt_name = str(config.get("optimizer", "adam")).lower()
    lr = float(config.get("lr", 5e-4))
    weight_decay = float(config.get("weight_decay", 1e-5))

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    raise ValueError("optimizer must be 'adam' or 'adamw'.")


def train_trial_model(
    model: nn.Module,
    train_loader,
    val_loader,
    config: Dict[str, Any],
    device: torch.device,
    trial: optuna.Trial,
) -> Tuple[nn.Module, Dict[str, List[float]], Dict[str, Any]]:
    """
    Train one Optuna trial.

    Pruning is based on validation latent loss because it is cheap and stable.
    Final Optuna objective is computed from decoded impedance validation subset.
    """
    criterion, loss_tag = _build_loss(config)
    optimizer = _build_optimizer(model, config)

    epochs = int(config.get("epochs", 500))
    patience = int(config.get("patience", 50))
    min_delta = float(config.get("min_delta", 1e-6))
    grad_clip = float(config.get("grad_clip", 1.0))

    model.to(device)

    best_state = None
    best_val = float("inf")
    best_epoch = None
    bad_epochs = 0
    stopped_epoch = None

    history = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    start = time.time()

    for epoch in range(1, epochs + 1):
        # ---------------- TRAIN ----------------
        model.train()
        train_sum = 0.0
        n_train = 0

        for xb, zb, _, _ in train_loader:
            xb = xb.to(device)
            zb = zb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, zb)
            loss.backward()

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

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["lr"].append(float(current_lr))

        # ---------------- BEST ----------------
        improved = (best_val - val_loss) > min_delta

        if improved:
            best_val = float(val_loss)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        # ---------------- OPTUNA PRUNING ----------------
        trial.report(val_loss, step=epoch)

        if trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at epoch {epoch} with val_loss={val_loss:.6f}")

        # ---------------- EARLY STOPPING ----------------
        if bad_epochs >= patience:
            stopped_epoch = int(epoch)
            break

    elapsed = time.time() - start

    if best_state is not None:
        model.load_state_dict(best_state)

    if stopped_epoch is None:
        stopped_epoch = len(history["val_loss"])

    times = {
        "train_time_sec": float(elapsed),
        "train_time_min": float(elapsed / 60.0),
        "best_epoch": int(best_epoch if best_epoch is not None else -1),
        "best_val_loss": float(best_val),
        "stopped_epoch": int(stopped_epoch),
        "early_stopped": int(stopped_epoch < epochs),
        "loss_tag": loss_tag,
    }

    return model, history, times


@torch.no_grad()
def evaluate_decoded_subset(
    model: nn.Module,
    subset_loader,
    z_scaler,
    ae: nn.Module,
    stats1: Dict[str, torch.Tensor],
    imp_cache: Dict[str, Any],
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate predicted latent vectors after CNN-AE decoder on a validation subset.

    Returns impedance-domain metrics:
        rmse_ohm, mae_ohm, nmae_pct, mae_db, mae_phase_deg, band_mae_db_*
    """
    model.eval()
    ae.eval()

    freq_hz = np.asarray(imp_cache["freq_hz"], dtype=np.float64)

    all_sids = imp_cache["simu_ids"].cpu().numpy().astype(int)
    sid_to_pos = {int(s): i for i, s in enumerate(all_sids)}

    z_mean = torch.as_tensor(z_scaler.mean_, dtype=torch.float32, device=device)
    z_scale = torch.as_tensor(z_scaler.scale_, dtype=torch.float32, device=device)

    y_true_batches = []
    y_pred_batches = []

    for xb, _, _, sids in subset_loader:
        xb = xb.to(device)

        # geometry -> normalized latent
        z_pred_norm = model(xb)

        # normalized latent -> physical latent
        z_pred = z_pred_norm * z_scale + z_mean

        # latent -> normalized decoder output
        y_pred_norm = ae.decode(z_pred)

        # denormalize decoder output -> physical ln(|Z|), phase
        y_pred_phys = y_pred_norm.clone()
        y_pred_phys[:, 0, :] = (
            y_pred_phys[:, 0, :] * (stats1["ch0_std"] + 1e-6)
            + stats1["ch0_mean"]
        )
        y_pred_phys[:, 1, :] = (
            y_pred_phys[:, 1, :] * (stats1["ch1_std"] + 1e-6)
            + stats1["ch1_mean"]
        )

        pos = [sid_to_pos[int(s)] for s in sids.numpy().astype(int)]
        y_true = imp_cache["impedance"][pos].cpu().numpy()

        y_true_batches.append(y_true)
        y_pred_batches.append(y_pred_phys.cpu().numpy())

    y_true_all = np.concatenate(y_true_batches, axis=0)
    y_pred_all = np.concatenate(y_pred_batches, axis=0)

    orig_ohm = np.exp(y_true_all[:, 0, :])
    pred_ohm = np.exp(y_pred_all[:, 0, :])

    orig_phs = y_true_all[:, 1, :]
    pred_phs = y_pred_all[:, 1, :]

    metrics = compute_metrics(
        orig_ohm=orig_ohm,
        pred_ohm=pred_ohm,
        freq_hz=freq_hz,
        orig_phs=orig_phs,
        pred_phs=pred_phs,
    )

    return metrics


def make_subset_loader(dataset, subset_size: int, batch_size: int, seed: int = 42):
    """
    Build a deterministic random subset loader from a validation dataset.
    """
    n = len(dataset)
    subset_size = int(min(subset_size, n))

    rng = np.random.default_rng(int(seed))
    subset_indices = rng.choice(np.arange(n), size=subset_size, replace=False).tolist()

    subset = Subset(dataset, subset_indices)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


def make_objective(
    base_cfg: Dict[str, Any],
    train_dataset,
    val_dataset,
    input_dim: int,
    latent_dim: int,
    z_scaler,
    ae: nn.Module,
    stats1: Dict[str, torch.Tensor],
    imp_cache: Dict[str, Any],
    device: torch.device,
):
    """
    Returns an Optuna objective function.

    The objective minimizes:
        val_mae_db + phase_weight * val_mae_phase_deg

    using a deterministic subset of validation data for decoder evaluation.
    """

    def objective(trial: optuna.Trial) -> float:
        set_global_seed(int(base_cfg.get("split_seed", 42)) + trial.number)

        trial_cfg = suggest_stage2_hparams(trial, base_cfg)

        batch_size = int(trial_cfg["batch_size"])

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        val_subset_loader = make_subset_loader(
            val_dataset,
            subset_size=int(base_cfg.get("optuna_val_decode_subset_size", 1024)),
            batch_size=batch_size,
            seed=int(base_cfg.get("split_seed", 42)),
        )

        model = build_stage2_model(
            model_type=trial_cfg["model_type"],
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=trial_cfg["hidden_dims"],
            activation=trial_cfg["activation"],
            dropout=trial_cfg["dropout"],
            use_batchnorm=trial_cfg["use_batchnorm"],
        )

        n_trainable = count_trainable_parameters(model)

        trial_run_name = f"trial_{trial.number:03d}"

        with mlflow.start_run(run_name=trial_run_name, nested=True):
            mlflow.set_tags(
                {
                    "trial_number": trial.number,
                    "trial_type": "optuna_child_run",
                    "task": "stage2_hyperparameter_tuning",
                }
            )

            mlflow.log_params(
                {
                    "trial_number": trial.number,
                    "architecture": trial_cfg["architecture"],
                    "hidden_dims": str(trial_cfg["hidden_dims"]),
                    "activation": trial_cfg["activation"],
                    "dropout": trial_cfg["dropout"],
                    "use_batchnorm": trial_cfg["use_batchnorm"],
                    "optimizer": trial_cfg["optimizer"],
                    "loss_function": trial_cfg["loss_function"],
                    "huber_delta": trial_cfg.get("huber_delta", None),
                    "lr": trial_cfg["lr"],
                    "weight_decay": trial_cfg["weight_decay"],
                    "batch_size": trial_cfg["batch_size"],
                    "grad_clip": trial_cfg["grad_clip"],
                    "epochs": trial_cfg["epochs"],
                    "patience": trial_cfg["patience"],
                    "trainable_parameters": n_trainable,
                }
            )

            try:
                model, history, times = train_trial_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    config=trial_cfg,
                    device=device,
                    trial=trial,
                )

                decoded_metrics = evaluate_decoded_subset(
                    model=model,
                    subset_loader=val_subset_loader,
                    z_scaler=z_scaler,
                    ae=ae,
                    stats1=stats1,
                    imp_cache=imp_cache,
                    device=device,
                )

                phase_weight = float(base_cfg.get("optuna_phase_weight", 0.02))

                objective_score = (
                    float(decoded_metrics["mae_db"])
                    + phase_weight * float(decoded_metrics["mae_phase_deg"])
                )

                mlflow.log_metric("objective_score", float(objective_score))
                mlflow.log_metric("val_best_latent_loss", float(times["best_val_loss"]))
                mlflow.log_metric("best_epoch", int(times["best_epoch"]))
                mlflow.log_metric("stopped_epoch", int(times["stopped_epoch"]))
                mlflow.log_metric("early_stopped", int(times["early_stopped"]))
                mlflow.log_metric("train_time_sec", float(times["train_time_sec"]))
                mlflow.log_metric("train_time_min", float(times["train_time_min"]))

                for k, v in decoded_metrics.items():
                    mlflow.log_metric(f"val_subset_{k}", float(v))

                trial.set_user_attr("objective_score", float(objective_score))
                trial.set_user_attr("val_subset_mae_db", float(decoded_metrics["mae_db"]))
                trial.set_user_attr("val_subset_mae_phase_deg", float(decoded_metrics["mae_phase_deg"]))
                trial.set_user_attr("best_val_latent_loss", float(times["best_val_loss"]))
                trial.set_user_attr("best_epoch", int(times["best_epoch"]))
                trial.set_user_attr("trainable_parameters", int(n_trainable))

                return float(objective_score)

            except optuna.TrialPruned:
                mlflow.set_tag("trial_status", "pruned")
                raise

            except Exception as e:
                mlflow.set_tag("trial_status", "failed")
                mlflow.log_param("failure_reason", str(e)[:200])
                raise

    return objective