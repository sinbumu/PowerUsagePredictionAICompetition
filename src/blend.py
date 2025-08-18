from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
	from scipy.optimize import nnls
	_HAS_NNLS = True
except Exception:
	_HAS_NNLS = False

from .metrics import smape


def simple_blend(preds: List[np.ndarray], weights: List[float]) -> np.ndarray:
	w = np.asarray(weights, dtype=float)
	w = w / (w.sum() + 1e-12)
	stack = np.vstack(preds)
	return (w[:, None] * stack).sum(axis=0)


def nnls_blend(oof_list: List[np.ndarray], y_true: np.ndarray) -> np.ndarray:
	if not _HAS_NNLS:
		raise RuntimeError("scipy not available for nnls blend")
	X = np.vstack(oof_list).T
	w, _ = nnls(X, y_true)
	if w.sum() == 0:
		w = np.ones_like(w)
	return w / w.sum()


def evaluate_blend(oof_list: List[np.ndarray], y_true: np.ndarray, method: str = "simple", weights: List[float] | None = None) -> Dict[str, float]:
	if method == "simple":
		assert weights is not None and len(weights) == len(oof_list)
		yhat = simple_blend(oof_list, weights)
		return {"smape": float(smape(y_true, yhat))}
	elif method == "nnls":
		w = nnls_blend(oof_list, y_true)
		yhat = simple_blend(oof_list, list(w))
		return {"smape": float(smape(y_true, yhat)), "weights": w.tolist()}
	else:
		raise ValueError("unknown blend method")


def find_weights_smape_grid(oof_list: List[np.ndarray], y_true: np.ndarray, step: float = 0.02) -> Tuple[np.ndarray, float]:
	"""Grid search non-negative weights on simplex to minimize SMAPE.
	Works efficiently for 2-3 models. For 2 models we scan w in [0,1].
	"""
	n = len(oof_list)
	if n == 1:
		return np.array([1.0]), float(smape(y_true, oof_list[0]))
	best_smape = float("inf")
	best_w = None
	if n == 2:
		w_vals = np.arange(0.0, 1.0 + 1e-9, step)
		p0, p1 = oof_list
		for w in w_vals:
			pred = w * p0 + (1.0 - w) * p1
			s = smape(y_true, pred)
			if s < best_smape:
				best_smape, best_w = s, np.array([w, 1.0 - w])
		return best_w, float(best_smape)
	# generic (3+): simple Dirichlet grid (coarse)
	grid = np.arange(0.0, 1.0 + 1e-9, step)
	for w0 in grid:
		for w1 in grid:
			rem = 1.0 - w0 - w1
			if n == 3:
				if rem < -1e-9:
					continue
				w = np.array([w0, w1, max(rem, 0.0)])
				w = w / (w.sum() + 1e-12)
				pred = simple_blend(oof_list, list(w))
				s = smape(y_true, pred)
				if s < best_smape:
					best_smape, best_w = s, w
			else:
				# fallback: equal weights
				w = np.ones(n) / n
				pred = simple_blend(oof_list, list(w))
				s = smape(y_true, pred)
				if s < best_smape:
					best_smape, best_w = s, w
	return best_w, float(best_smape)
