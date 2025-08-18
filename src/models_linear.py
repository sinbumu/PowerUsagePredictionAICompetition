from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import EPS, MODELS_DIR
from .metrics import smape
from .features import FeatureBuilder, TIME_FEATURE_COLUMNS, WEATHER_NOW_COLUMNS
from .cv import BlockTimeSeriesSplit
from .baseline import build_lag_features as build_lag_for_test


@dataclass
class ResidualRidgeModel:
	feature_names: List[str]
	means: np.ndarray
	stds: np.ndarray
	weights: np.ndarray
	bias: float
	alpha: float = 1.0

	def predict(self, X: pd.DataFrame) -> np.ndarray:
		X = X[self.feature_names].values.astype(float)
		Xs = (X - self.means) / (self.stds + 1e-8)
		return Xs @ self.weights + self.bias

	def save(self, path: str) -> None:
		os.makedirs(os.path.dirname(path), exist_ok=True)
		np.savez(path, feature_names=np.array(self.feature_names), means=self.means, stds=self.stds, weights=self.weights, bias=self.bias, alpha=self.alpha)

	@staticmethod
	def load(path: str) -> "ResidualRidgeModel":
		data = np.load(path, allow_pickle=True)
		return ResidualRidgeModel(
			feature_names=list(data["feature_names"].tolist()),
			means=data["means"],
			stds=data["stds"],
			weights=data["weights"],
			bias=float(data["bias"]),
			alpha=float(data["alpha"]),
		)


def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
	# standardize
	means = X.mean(axis=0)
	stds = X.std(axis=0)
	stds[stds == 0] = 1.0
	Xs = (X - means) / stds
	# closed-form ridge: (X^T X + alpha I) w = X^T y
	n_features = Xs.shape[1]
	A = Xs.T @ Xs + alpha * np.eye(n_features)
	b = Xs.T @ y
	w = np.linalg.solve(A, b)
	# bias as mean(y - Xw)
	bias = float(y.mean() - Xs.mean(axis=0) @ w)
	return w, bias, means, stds


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
	cols: List[str] = []
	cols += TIME_FEATURE_COLUMNS
	# weather now, lag168, diff168
	for c in WEATHER_NOW_COLUMNS:
		if c in df.columns:
			cols.append(c)
			if f"{c}_lag168" in df.columns:
				cols.append(f"{c}_lag168")
			if f"{c}_diff168" in df.columns:
				cols.append(f"{c}_diff168")
	# load lag for residual support
	if "lag_168" in df.columns:
		cols.append("lag_168")
	# rolling 168 of load if available
	if "roll_mean_168" in df.columns:
		cols.append("roll_mean_168")
	# meta numeric columns (optional)
	for c in ["total_area", "cooling_area", "pv_capacity", "ess_capacity", "pcs_capacity"]:
		if c in df.columns:
			cols.append(c)
	# drop duplicates while preserving order
	seen = set()
	ordered = []
	for c in cols:
		if c not in seen and c in df.columns:
			seen.add(c)
			ordered.append(c)
	return ordered


def train_residual_model(train_df: pd.DataFrame, building_info: pd.DataFrame, alpha: float = 1.0, save_dir: str = os.path.join(MODELS_DIR, "linear_resid")) -> Dict[str, float]:
	fb = FeatureBuilder()
	train_feat, _ = fb.transform(train_df, building_info=building_info)
	# target residual
	if "lag_168" not in train_feat.columns:
		raise RuntimeError("lag_168 not computed in features; check FeatureBuilder")
	train_feat["residual"] = train_feat["load"] - train_feat["lag_168"]
	# drop rows without lag_168
	train_feat = train_feat.dropna(subset=["residual"]) 
	# select features
	feat_cols = _select_feature_columns(train_feat)
	X_all = train_feat[feat_cols].fillna(0.0).values.astype(float)
	y_all = train_feat["residual"].values.astype(float)
	# CV
	splitter = BlockTimeSeriesSplit(block_hours=168, gap_hours=24, min_train_blocks=1)
	fold_metrics = []
	train_feat = train_feat.reset_index(drop=True)
	OOF = np.full(len(train_feat), np.nan)
	for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(train_feat)):
		w, b, m, s = _ridge_fit(X_all[tr_idx], y_all[tr_idx], alpha=alpha)
		model = ResidualRidgeModel(feat_cols, m, s, w, b, alpha)
		rhat = model.predict(train_feat.iloc[va_idx][feat_cols])
		yhat = np.clip(train_feat.iloc[va_idx]["lag_168"].values + rhat, 0.0, None)
		fold_smape = smape(train_feat.iloc[va_idx]["load"].values, yhat)
		fold_metrics.append(fold_smape)
		OOF[va_idx] = yhat
	# report on valid predictions only
	mask = ~np.isnan(OOF)
	oof_smape = smape(train_feat.loc[mask, "load"].values, OOF[mask])
	os.makedirs(save_dir, exist_ok=True)
	with open(os.path.join(save_dir, "cv_metrics.json"), "w", encoding="utf-8") as f:
		json.dump({"fold_smape": fold_metrics, "oof_smape": float(oof_smape)}, f, ensure_ascii=False, indent=2)
	# save OOF
	oof_df = train_feat.loc[mask, ["building_id", "timestamp", "load"]].copy()
	oof_df["oof_pred"] = OOF[mask]
	oof_df.to_csv(os.path.join(save_dir, "oof.csv"), index=False)
	# final fit
	w, b, m, s = _ridge_fit(X_all, y_all, alpha=alpha)
	final_model = ResidualRidgeModel(feat_cols, m, s, w, b, alpha)
	final_model.save(os.path.join(save_dir, "model.npz"))
	return {"oof_smape": float(oof_smape), "fold_smape_mean": float(np.mean(fold_metrics))}


def infer_residual_model(train_df: pd.DataFrame, test_df: pd.DataFrame, building_info: pd.DataFrame, model_path: str) -> pd.DataFrame:
	# compute lag_168 for test from train history
	_, test_lag = build_lag_for_test(train_df, test_df)
	test_lag = test_lag.rename(columns={"lag168": "lag_168"})
	# build weather/time/meta features using concatenation of train+test (to allow weather lag168)
	use_cols = ["building_id", "timestamp", "temp", "rain", "wind", "humid"]
	train_subset = train_df[[c for c in use_cols if c in train_df.columns]].copy()
	test_subset = test_df[[c for c in use_cols if c in test_df.columns]].copy()
	train_subset["is_test"] = 0
	test_subset["is_test"] = 1
	combo = pd.concat([train_subset, test_subset], ignore_index=True)
	fb = FeatureBuilder()
	combo_feat, _ = fb.transform(combo, building_info=building_info)
	# slice test part by flag (since transform sorts globally)
	test_feat = combo_feat[combo_feat["is_test"] == 1].reset_index(drop=True)
	# attach lag_168 from history
	test_feat = test_feat.merge(test_lag[["building_id", "timestamp", "lag_168", "roll_mean_168"]], on=["building_id", "timestamp"], how="left")
	# select features
	model = ResidualRidgeModel.load(model_path)
	X = test_feat[model.feature_names].fillna(0.0)
	# predict residual and reconstruct load
	rhat = model.predict(X)
	yhat = np.clip(test_feat["lag_168"].values + rhat, 0.0, None)
	out = pd.DataFrame({
		"building_id": test_feat["building_id"].values,
		"timestamp": test_feat["timestamp"].values,
		"pred": yhat,
	})
	return out
