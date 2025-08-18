from __future__ import annotations

import numpy as np


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
	"""Compute Symmetric MAPE in percent scale (0-200).

	Args:
		y_true: ground truth (array-like)
		y_pred: predictions (array-like)
		eps: small constant to avoid division by zero
	"""
	y_true = np.asarray(y_true, dtype=float)
	y_pred = np.asarray(y_pred, dtype=float)
	num = np.abs(y_true - y_pred)
	den = np.abs(y_true) + np.abs(y_pred) + eps
	return 200.0 * np.mean(num / den)
