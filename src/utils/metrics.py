"""
Traffic Prediction Evaluation Metrics
=======================================
Standard metrics used in traffic forecasting benchmarks:
  - MAE  : Mean Absolute Error
  - RMSE : Root Mean Squared Error
  - MAPE : Mean Absolute Percentage Error

All metrics support a null_val mask to ignore missing sensor readings (0 or NaN).
"""

import numpy as np
import torch
from typing import Union

Array = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: Array) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _mask(pred: np.ndarray, target: np.ndarray,
          null_val: float = np.nan) -> np.ndarray:
    """Return boolean mask of valid (non-null) target entries."""
    if np.isnan(null_val):
        return ~np.isnan(target)
    return target != null_val


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def masked_mae(pred: Array, target: Array, null_val: float = np.nan) -> float:
    """Mean Absolute Error, ignoring null_val entries in target."""
    pred, target = _to_numpy(pred), _to_numpy(target)
    mask = _mask(pred, target, null_val)
    if mask.sum() == 0:
        return 0.0
    return float(np.abs(pred[mask] - target[mask]).mean())


def masked_rmse(pred: Array, target: Array, null_val: float = np.nan) -> float:
    """Root Mean Squared Error, ignoring null_val entries."""
    pred, target = _to_numpy(pred), _to_numpy(target)
    mask = _mask(pred, target, null_val)
    if mask.sum() == 0:
        return 0.0
    return float(np.sqrt(((pred[mask] - target[mask]) ** 2).mean()))


def masked_mape(pred: Array, target: Array, null_val: float = np.nan,
                epsilon: float = 1e-5) -> float:
    """Mean Absolute Percentage Error (%) ignoring near-zero / null targets."""
    pred, target = _to_numpy(pred), _to_numpy(target)
    mask = _mask(pred, target, null_val) & (np.abs(target) > epsilon)
    if mask.sum() == 0:
        return 0.0
    return float(100.0 * np.abs((pred[mask] - target[mask]) / target[mask]).mean())


def masked_mse(pred: Array, target: Array, null_val: float = np.nan) -> float:
    """Mean Squared Error, ignoring null_val entries."""
    pred, target = _to_numpy(pred), _to_numpy(target)
    mask = _mask(pred, target, null_val)
    if mask.sum() == 0:
        return 0.0
    return float(((pred[mask] - target[mask]) ** 2).mean())


# ---------------------------------------------------------------------------
# Horizon-specific metrics
# ---------------------------------------------------------------------------

def metrics_by_horizon(pred: Array, target: Array,
                       null_val: float = np.nan,
                       horizons=(3, 6, 12)) -> dict:
    """
    Compute MAE / RMSE / MAPE for specific forecast horizons.

    Args:
        pred    : (B, T_out, N) or (B, T_out, N, F) predictions
        target  : same shape as pred
        horizons: tuple of 1-based horizon indices to evaluate

    Returns:
        dict mapping horizon -> {'MAE': ..., 'RMSE': ..., 'MAPE': ...}
    """
    pred, target = _to_numpy(pred), _to_numpy(target)
    results = {}
    for h in horizons:
        if h > pred.shape[1]:
            continue
        p_h = pred[:, h - 1]
        t_h = target[:, h - 1]
        results[h] = {
            'MAE': masked_mae(p_h, t_h, null_val),
            'RMSE': masked_rmse(p_h, t_h, null_val),
            'MAPE': masked_mape(p_h, t_h, null_val),
        }
    return results


def print_metrics_table(results: dict, dataset: str = ''):
    """Pretty-print a horizon metrics table."""
    header = f'{"Horizon":>10} {"MAE":>10} {"RMSE":>10} {"MAPE":>10}'
    sep = '-' * len(header)
    if dataset:
        print(f'\nDataset: {dataset}')
    print(sep)
    print(header)
    print(sep)
    for h, m in sorted(results.items()):
        print(f'{f"{h*5}min":>10} {m["MAE"]:>10.4f} {m["RMSE"]:>10.4f} {m["MAPE"]:>9.2f}%')
    print(sep)


# ---------------------------------------------------------------------------
# Torch-compatible loss wrappers (used inside training loop)
# ---------------------------------------------------------------------------

class MaskedMAELoss(torch.nn.Module):
    """MAE loss that zeros out null_val targets (e.g. 0.0 for missing sensors)."""

    def __init__(self, null_val: float = 0.0):
        super().__init__()
        self.null_val = null_val

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if np.isnan(self.null_val):
            mask = ~torch.isnan(target)
        else:
            mask = target != self.null_val
        mask = mask.float()
        loss = torch.abs(pred - target) * mask
        return loss.sum() / mask.sum().clamp(min=1)


class MaskedRMSELoss(torch.nn.Module):
    """RMSE loss with null_val masking."""

    def __init__(self, null_val: float = 0.0):
        super().__init__()
        self.null_val = null_val

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if np.isnan(self.null_val):
            mask = ~torch.isnan(target)
        else:
            mask = target != self.null_val
        mask = mask.float()
        loss = ((pred - target) ** 2) * mask
        return torch.sqrt(loss.sum() / mask.sum().clamp(min=1))
