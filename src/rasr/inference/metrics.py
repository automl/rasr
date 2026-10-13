from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score as _sklearn_r2_score


def _manual_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("-inf")
    return 1.0 - ss_res / ss_tot


def nrmse_reward(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes 1 / (1 + NRMSE) where NRMSE = RMSE / std(y_true). Range: (0, 1]."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))):
        return 0.0
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    std = float(np.std(y_true))
    if std == 0.0:
        return 1.0 if rmse == 0.0 else 0.0
    return 1.0 / (1.0 + rmse / std)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("shape mismatch")
    if not (np.all(np.isfinite(y_true)) and np.all(np.isfinite(y_pred))):
        return float("-inf")

    try:
        r2 = _sklearn_r2_score(y_true, y_pred)
    except Exception:
        r2 = _manual_r2(y_true, y_pred)

    return min(r2, 1.0)
