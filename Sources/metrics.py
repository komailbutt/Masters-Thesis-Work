# Sources/metrics.py
# Reusable metrics for Stage-1 and Stage-2 evaluation.
#
# Why a separate file?
# - You will use the exact same evaluation metrics after stage-2 (geo -> latent -> decoder -> impedance),
#   so keeping it in Sources/metrics.py avoids copy-paste.
#
# Notes:
# - Magnitude metrics are computed on |Z| in Ohms.
# - dB metrics use 20*log10(|Z|) which is standard for impedance magnitude plots.
# - Phase MAE must handle wrap-around (e.g., 359° and 1° are close).

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np


def mag_to_db(mag_ohm: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Convert magnitude (Ohms) to dB scale using 20*log10(|Z|).
    """
    mag = np.maximum(mag_ohm, eps)
    return 20.0 * np.log10(mag)


def phase_mae_deg(orig_phs: np.ndarray, pred_phs: np.ndarray) -> float:
    """
    Mean absolute phase error in degrees with circular wrap handling.

    Args:
        orig_phs, pred_phs: arrays of phase in radians, shape [N, Nf] (or broadcastable)

    Returns:
        mae_phase_deg: scalar float
    """
    if orig_phs.shape != pred_phs.shape:
        raise ValueError(f"Phase shape mismatch: orig {orig_phs.shape} vs pred {pred_phs.shape}")

    diff_rad = orig_phs - pred_phs
    # Wrap to [-pi, pi] to measure shortest angular distance
    diff_rad = (diff_rad + np.pi) % (2 * np.pi) - np.pi
    return float(np.mean(np.abs(np.degrees(diff_rad))))


def compute_banded_mae_db(
    orig_db: np.ndarray,
    pred_db: np.ndarray,
    freq_hz: np.ndarray,
    bands: Optional[List[Tuple[str, float, float]]] = None,
) -> Dict[str, float]:
    """
    Compute banded MAE in dB.

    Args:
        orig_db, pred_db: [N, Nf] arrays in dB
        freq_hz: [Nf] frequency vector in Hz
        bands: list of (name, f0_hz, f1_hz). If None -> default: low/mid/high (matches Cell 12)

    Returns:
        dict: {"band_mae_db_low": value, ...}
    """
    if bands is None:
        # Matches your Cell 12 bands exactly (in Hz)
        bands = [
            ("low",  1e6,   10e6),
            ("mid",  10e6,  100e6),
            ("high", 100e6, 1e9 + 1),
        ]

    out: Dict[str, float] = {}
    for name, f0, f1 in bands:
        mask = (freq_hz >= f0) & (freq_hz < f1)
        if not np.any(mask):
            out[f"band_mae_db_{name}"] = float("nan")
            continue
        out[f"band_mae_db_{name}"] = float(np.mean(np.abs(orig_db[:, mask] - pred_db[:, mask])))

    return out


def compute_metrics(
    orig_ohm: np.ndarray,
    pred_ohm: np.ndarray,
    freq_hz: np.ndarray,
    orig_phs: Optional[np.ndarray] = None,
    pred_phs: Optional[np.ndarray] = None,
    eps: float = 1e-12,
    bands: Optional[List[Tuple[str, float, float]]] = None,
) -> Dict[str, float]:
    """
    Computes the same metrics as your Cell 12:
      - rmse_ohm, mae_ohm, nmae_pct
      - mae_db
      - mae_phase_deg (if phase is provided)
      - band_mae_db_low/mid/high

    Args:
        orig_ohm, pred_ohm: [N, Nf] magnitude in Ohms
        freq_hz: [Nf] Hz
        orig_phs, pred_phs: [N, Nf] phase in radians (optional)
        bands: optional custom bands [(name,f0_hz,f1_hz), ...]

    Returns:
        dict of metrics
    """
    if orig_ohm.shape != pred_ohm.shape:
        raise ValueError(f"Shape mismatch: orig {orig_ohm.shape} vs pred {pred_ohm.shape}")
    if orig_ohm.ndim != 2:
        raise ValueError("Expected orig_ohm and pred_ohm to be 2D arrays [N, Nf].")
    if freq_hz.ndim != 1 or freq_hz.shape[0] != orig_ohm.shape[1]:
        raise ValueError("freq_hz must be [Nf] and match second dimension of orig_ohm/pred_ohm.")

    rmse_ohm = float(np.sqrt(np.mean((orig_ohm - pred_ohm) ** 2)))
    mae_ohm = float(np.mean(np.abs(orig_ohm - pred_ohm)))
    nmae_pct = float(mae_ohm / (np.mean(orig_ohm) + eps) * 100.0)

    orig_db = 20.0 * np.log10(orig_ohm + eps)
    pred_db = 20.0 * np.log10(pred_ohm + eps)
    mae_db = float(np.mean(np.abs(orig_db - pred_db)))

    metrics: Dict[str, float] = {
        "rmse_ohm": rmse_ohm,
        "mae_ohm": mae_ohm,
        "nmae_pct": nmae_pct,
        "mae_db": mae_db,
    }

    # Phase metric (optional)
    if (orig_phs is not None) and (pred_phs is not None):
        metrics["mae_phase_deg"] = phase_mae_deg(orig_phs, pred_phs)

    # Banded metrics in dB
    metrics.update(compute_banded_mae_db(orig_db, pred_db, freq_hz, bands=bands))
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "CNN AE Performance") -> None:
    """
    Pretty print metrics in the same spirit as your Cell 12 output.
    """
    print(title)
    if "rmse_ohm" in metrics: print(f"RMSE (Ohm)   : {metrics['rmse_ohm']:.6f} Ω")
    if "mae_ohm" in metrics:  print(f"MAE  (Ohm)   : {metrics['mae_ohm']:.6f} Ω")
    if "nmae_pct" in metrics: print(f"nMAE (%)     : {metrics['nmae_pct']:.4f} %")
    if "mae_db" in metrics:   print(f"MAE  (dB)    : {metrics['mae_db']:.6f} dB")
    if "mae_phase_deg" in metrics: print(f"MAE  (Phase) : {metrics['mae_phase_deg']:.4f} deg")

    # band metrics (if present)
    band_keys = [k for k in metrics.keys() if k.startswith("band_mae_db_")]
    if band_keys:
        print("\nBanded MAE (dB):")
        for k in sorted(band_keys):
            name = k.replace("band_mae_db_", "")
            print(f"  {name:>4s}: {metrics[k]:.6f} dB")
