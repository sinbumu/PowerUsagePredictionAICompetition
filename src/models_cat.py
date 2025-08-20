from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import dump, load

from .config import MODELS_DIR
from .metrics import smape
from .features import FeatureBuilder
from .cv import BlockTimeSeriesSplit, last_weeks_splits
from .baseline import build_lag_features as build_lag_for_test

try:
	import catboost as cb  # type: ignore
	_HAS_CAT = True
except Exception:
	_HAS_CAT = False

from sklearn.ensemble import HistGradientBoostingRegressor


def _select_features(df: pd.DataFrame, allow_cat: bool) -> List[str]:
	# Only inference-reproducible features; include allowed categoricals for CatBoost
	exclude = {"load", "residual", "timestamp", "num_date_time"}
	allowed_cat = {"building_id", "type"} if allow_cat else set()
	cols: List[str] = []
	for c in df.columns:
		if c in exclude:
			continue
		if c.startswith("lag_") and c != "lag_168":
			continue
		if c.startswith("roll_mean_") and c != "roll_mean_168":
			continue
		if c.startswith("roll_std_"):
			continue
		if c in allowed_cat:
			cols.append(c)
			continue
		if pd.api.types.is_numeric_dtype(df[c]):
			cols.append(c)
	return cols


def _hour_of_week(ts: pd.Series) -> pd.Series:
	return ((ts.dt.dayofweek * 24) + ts.dt.hour).astype(int)


def train_cat_residual(train_df: pd.DataFrame, building_info: pd.DataFrame, save_dir: str = os.path.join(MODELS_DIR, "cat_resid"), params: Optional[Dict] = None, use_last_weeks_cv: bool = True, smape_weight_c: float = 200.0, save_bias: bool = True) -> Dict[str, float]:
	os.makedirs(save_dir, exist_ok=True)
	fb = FeatureBuilder()
	fe, _ = fb.transform(train_df, building_info=building_info)
	if "lag_168" not in fe.columns:
		raise RuntimeError("lag_168 missing in features")
	fe["residual"] = fe["load"] - fe["lag_168"]
	fe = fe.dropna(subset=["residual"]).reset_index(drop=True)

	feat_cols = _select_features(fe, allow_cat=_HAS_CAT)
	cat_cols = [c for c in ["building_id", "type"] if c in feat_cols and c in fe.columns]
	X = fe[feat_cols].copy()
	y = fe["residual"].values

	# sample weights to approximate SMAPE (down-weight large |y|)
	w = 1.0 / (np.abs(fe["load"].values) + smape_weight_c)

	if not _HAS_CAT:
		for c in cat_cols:
			X[c] = X[c].astype("category").cat.codes

	# CV: last 3 weeks anchored, else rolling
	folds = last_weeks_splits(fe, weeks=3, gap_hours=24) if use_last_weeks_cv else list(BlockTimeSeriesSplit().split(fe))
	OOF = np.full(len(fe), np.nan)
	fold_scores: List[float] = []

	default_params = {
		"loss_function": "MAE",
		"depth": 8,
		"learning_rate": 0.05,
		"l2_leaf_reg": 3.0,
		"iterations": 2000,
		"early_stopping_rounds": 200,
	}
	if params:
		default_params.update(params)

	models = []
	for fold_idx, (tr_idx, va_idx) in enumerate(folds):
		Xtr, ytr, wtr = X.iloc[tr_idx], y[tr_idx], w[tr_idx]
		Xva, yva = X.iloc[va_idx], y[va_idx]
		if _HAS_CAT:
			cat_indices = [X.columns.get_loc(c) for c in cat_cols]
			model = cb.CatBoostRegressor(**default_params, verbose=False)
			model.fit(Xtr, ytr, sample_weight=wtr, eval_set=(Xva, yva), cat_features=cat_indices)
			yva_res = model.predict(Xva)
		else:
			model = HistGradientBoostingRegressor(max_depth=None, max_iter=800, learning_rate=0.05)
			model.fit(Xtr, ytr, sample_weight=wtr)
			yva_res = model.predict(Xva)
		OOF[va_idx] = yva_res + fe.loc[va_idx, "lag_168"].values
		fold_scores.append(smape(fe.loc[va_idx, "load"].values, np.clip(OOF[va_idx], 0.0, None)))
		models.append(model)

	mask = ~np.isnan(OOF)
	oof_smape = smape(fe.loc[mask, "load"].values, np.clip(OOF[mask], 0.0, None))
	with open(os.path.join(save_dir, "cv_metrics.json"), "w", encoding="utf-8") as f:
		json.dump({"fold_smape": fold_scores, "oof_smape": float(oof_smape)}, f, ensure_ascii=False, indent=2)
	# save OOF
	oof_df = fe.loc[mask, ["building_id", "timestamp", "load"]].copy()
	oof_df["oof_pred"] = np.clip(OOF[mask], 0.0, None)
	oof_df.to_csv(os.path.join(save_dir, "oof.csv"), index=False)

	# bias correction: building_id x hour_of_week
	bias_path = os.path.join(save_dir, "bias_bh.json")
	if save_bias:
		bh = _hour_of_week(oof_df["timestamp"])
		bias_tbl = oof_df.assign(how=bh.values)
		bias_tbl["err"] = bias_tbl["oof_pred"] - bias_tbl["load"]
		d = bias_tbl.groupby(["building_id", "how"])["err"].mean().to_dict()
		with open(bias_path, "w", encoding="utf-8") as f:
			json.dump({f"{k[0]}_{k[1]}": float(v) for k, v in d.items()}, f)

	# final fit
	if _HAS_CAT:
		cat_indices = [X.columns.get_loc(c) for c in cat_cols]
		final = cb.CatBoostRegressor(**default_params, verbose=False)
		final.fit(X, y, sample_weight=w, cat_features=cat_indices)
		final_path = os.path.join(save_dir, "model.cbm")
		final.save_model(final_path)
	else:
		final = HistGradientBoostingRegressor(max_depth=None, max_iter=1200, learning_rate=0.05)
		final.fit(X, y, sample_weight=w)
		final_path = os.path.join(save_dir, "model_hgb.pkl")
		dump(final, final_path)

	with open(os.path.join(save_dir, "feature_names.json"), "w", encoding="utf-8") as f:
		json.dump({"feature_names": feat_cols, "categoricals": cat_cols}, f)

	return {"oof_smape": float(oof_smape)}


def infer_cat_residual(train_df: pd.DataFrame, test_df: pd.DataFrame, building_info: pd.DataFrame, model_dir: str) -> pd.DataFrame:
	# compute lag_168 & roll_mean_168 for test from train history
	_, test_hist = build_lag_for_test(train_df, test_df)
	test_hist = test_hist.rename(columns={"lag168": "lag_168"})
	# build features by concat for weather lagging
	use_cols = ["building_id", "timestamp", "temp", "rain", "wind", "humid", "sunshine", "irradiance", "type"]
	train_subset = train_df[[c for c in use_cols if c in train_df.columns]].copy(); train_subset["is_test"]=0
	test_subset = test_df[[c for c in use_cols if c in test_df.columns]].copy(); test_subset["is_test"]=1
	combo = pd.concat([train_subset, test_subset], ignore_index=True)
	fb = FeatureBuilder()
	combo_feat, _ = fb.transform(combo, building_info=building_info)
	test_feat = combo_feat[combo_feat["is_test"]==1].copy().reset_index(drop=True)
	# attach lag history
	test_feat = test_feat.merge(test_hist[["building_id","timestamp","lag_168","roll_mean_168"]], on=["building_id","timestamp"], how="left")

	with open(os.path.join(model_dir, "feature_names.json"), "r", encoding="utf-8") as f:
		meta = json.load(f)
	feat_cols = meta["feature_names"]
	cat_cols = meta.get("categoricals", [])
	bias_path = os.path.join(model_dir, "bias_bh.json")
	bias_tbl: Dict[str, float] = {}
	if os.path.exists(bias_path):
		bias_tbl = json.load(open(bias_path, "r", encoding="utf-8"))

	X = test_feat[feat_cols].copy()
	if _HAS_CAT:
		model_path = os.path.join(model_dir, "model.cbm")
		if os.path.exists(model_path):
			import catboost as cb
			model = cb.CatBoostRegressor()
			model.load_model(model_path)
			y_res = model.predict(X)
		else:
			for c in cat_cols:
				X[c] = X[c].astype("category").cat.codes
			model = load(os.path.join(model_dir, "model_hgb.pkl"))
			y_res = model.predict(X)
	else:
		for c in cat_cols:
			X[c] = X[c].astype("category").cat.codes
		model = load(os.path.join(model_dir, "model_hgb.pkl"))
		y_res = model.predict(X)

	yhat = np.clip(test_feat["lag_168"].values + y_res, 0.0, None)
	# bias correction apply
	if bias_tbl:
		how = _hour_of_week(test_feat["timestamp"]).values
		bid = test_feat["building_id"].values
		adj = np.array([bias_tbl.get(f"{int(b)}_{int(h)}", 0.0) for b, h in zip(bid, how)])
		yhat = np.clip(yhat - adj, 0.0, None)

	return pd.DataFrame({
		"building_id": test_feat["building_id"].values,
		"timestamp": test_feat["timestamp"].values,
		"pred": yhat,
	})
