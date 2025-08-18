from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import dump, load

from .config import MODELS_DIR
from .metrics import smape
from .features import FeatureBuilder
from .cv import BlockTimeSeriesSplit
from .baseline import build_lag_features as build_lag_for_test

try:
	import lightgbm as lgb
	_HAS_LGBM = True
except Exception:
	_HAS_LGBM = False

from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class ResidualGBMArtifacts:
	feature_names: List[str]
	model_path: str
	cv_report_path: str


def _select_features(df: pd.DataFrame) -> List[str]:
	# Use only features reproducible at inference time
	exclude = {"load", "residual", "timestamp"}
	cols = []
	for c in df.columns:
		if c in exclude:
			continue
		if not pd.api.types.is_numeric_dtype(df[c]):
			continue
		# Exclude load-derived features that we cannot build for test
		if c.startswith("lag_") and c != "lag_168":
			continue
		if c.startswith("roll_mean_") and c != "roll_mean_168":
			continue
		if c.startswith("roll_std_"):
			continue
		cols.append(c)
	return cols


def train_gbm_residual(train_df: pd.DataFrame, building_info: pd.DataFrame, save_dir: str = os.path.join(MODELS_DIR, "gbm_resid"), params: Optional[Dict] = None) -> Dict[str, float]:
	os.makedirs(save_dir, exist_ok=True)
	fb = FeatureBuilder()
	fe, _ = fb.transform(train_df, building_info=building_info)
	if "lag_168" not in fe.columns:
		raise RuntimeError("lag_168 missing in features")
	fe["residual"] = fe["load"] - fe["lag_168"]
	fe = fe.dropna(subset=["residual"]) 
	fe = fe.reset_index(drop=True)
	feat_cols = _select_features(fe)
	X = fe[feat_cols]
	y = fe["residual"].values

	splitter = BlockTimeSeriesSplit(block_hours=168, gap_hours=24, min_train_blocks=1)
	OOF = np.full(len(fe), np.nan)
	fold_scores: List[float] = []

	default_params = {
		"learning_rate": 0.05,
		"max_depth": -1,
		"num_leaves": 64,
		"subsample": 0.8,
		"colsample_bytree": 0.8,
		"n_estimators": 2000,
		"objective": "mae",
	}
	if params:
		default_params.update(params)

	for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(fe)):
		Xtr, ytr = X.iloc[tr_idx], y[tr_idx]
		Xva, yva = X.iloc[va_idx], y[va_idx]
		if _HAS_LGBM:
			model = lgb.LGBMRegressor(**default_params)
			model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="mae", callbacks=[lgb.early_stopping(200, verbose=False)])
			yva_res = model.predict(Xva)
		else:
			model = HistGradientBoostingRegressor(max_depth=None, max_iter=600, learning_rate=0.05, l2_regularization=0.0)
			model.fit(Xtr, ytr)
			yva_res = model.predict(Xva)
		OOF[va_idx] = yva_res + fe.loc[va_idx, "lag_168"].values
		fold_s = smape(fe.loc[va_idx, "load"].values, np.clip(OOF[va_idx], 0.0, None))
		fold_scores.append(fold_s)

	mask = ~np.isnan(OOF)
	oof_smape = smape(fe.loc[mask, "load"].values, np.clip(OOF[mask], 0.0, None))
	with open(os.path.join(save_dir, "cv_metrics.json"), "w", encoding="utf-8") as f:
		json.dump({"fold_smape": fold_scores, "oof_smape": float(oof_smape)}, f, ensure_ascii=False, indent=2)
	# save OOF
	oof_df = fe.loc[mask, ["building_id", "timestamp", "load"]].copy()
	oof_df["oof_pred"] = np.clip(OOF[mask], 0.0, None)
	oof_df.to_csv(os.path.join(save_dir, "oof.csv"), index=False)

	# Fit final on all data
	if _HAS_LGBM:
		final = lgb.LGBMRegressor(**default_params)
		final.fit(X, y)
		final_path = os.path.join(save_dir, "model_lgbm.pkl")
	else:
		final = HistGradientBoostingRegressor(max_depth=None, max_iter=800, learning_rate=0.05, l2_regularization=0.0)
		final.fit(X, y)
		final_path = os.path.join(save_dir, "model_hgb.pkl")
	# Save model and features
	dump(final, final_path)
	with open(os.path.join(save_dir, "feature_names.json"), "w", encoding="utf-8") as f:
		json.dump({"feature_names": feat_cols}, f)

	return {"oof_smape": float(oof_smape), "fold_smape_mean": float(np.mean(fold_scores))}


def infer_gbm_residual(train_df: pd.DataFrame, test_df: pd.DataFrame, building_info: pd.DataFrame, model_dir: str) -> pd.DataFrame:
	# build lag history for test
	_, test_hist = build_lag_for_test(train_df, test_df)
	test_hist = test_hist.rename(columns={"lag168": "lag_168"})
	# build features for test (concat route to allow weather lag)
	use_cols = ["building_id", "timestamp", "temp", "rain", "wind", "humid", "sunshine", "irradiance"]
	train_subset = train_df[[c for c in use_cols if c in train_df.columns]].copy(); train_subset["is_test"]=0
	test_subset = test_df[[c for c in use_cols if c in test_df.columns]].copy(); test_subset["is_test"]=1
	combo = pd.concat([train_subset, test_subset], ignore_index=True)
	fb = FeatureBuilder()
	combo_feat, _ = fb.transform(combo, building_info=building_info)
	test_feat = combo_feat[combo_feat["is_test"]==1].copy().reset_index(drop=True)
	# attach lag_168 and roll_mean_168 from history
	test_feat = test_feat.merge(test_hist[["building_id","timestamp","lag_168","roll_mean_168"]], on=["building_id","timestamp"], how="left")
	# load model
	with open(os.path.join(model_dir, "feature_names.json"), "r", encoding="utf-8") as f:
		feat_cols = json.load(f)["feature_names"]
	model_path = os.path.join(model_dir, "model_lgbm.pkl")
	if not os.path.exists(model_path):
		model_path = os.path.join(model_dir, "model_hgb.pkl")
	model = load(model_path)
	yhat_res = model.predict(test_feat[feat_cols].fillna(0.0))
	yhat = np.clip(test_feat["lag_168"].values + yhat_res, 0.0, None)
	return pd.DataFrame({
		"building_id": test_feat["building_id"].values,
		"timestamp": test_feat["timestamp"].values,
		"pred": yhat,
	})
