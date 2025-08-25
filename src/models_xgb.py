from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import dump, load

from .config import MODELS_DIR
from .metrics import smape
from .features import FeatureBuilder
from .cv import last_weeks_splits
from .baseline import build_lag_features as build_lag_for_test

import xgboost as xgb


def _select_features(df: pd.DataFrame) -> List[str]:
	exclude = {"load", "residual", "timestamp"}
	cols = []
	for c in df.columns:
		if c in exclude:
			continue
		if not pd.api.types.is_numeric_dtype(df[c]):
			continue
		# allow only inference-available load features
		if c.startswith("lag_") and c != "lag_168":
			continue
		if c.startswith("roll_mean_") and c != "roll_mean_168":
			continue
		if c.startswith("roll_std_"):
			continue
		cols.append(c)
	return cols


def train_xgb_residual(train_df: pd.DataFrame, building_info: pd.DataFrame, save_dir: str = os.path.join(MODELS_DIR, "xgb_resid"), params: Optional[Dict] = None) -> Dict[str, float]:
	os.makedirs(save_dir, exist_ok=True)
	fb = FeatureBuilder()
	fe, _ = fb.transform(train_df, building_info=building_info)
	fe["residual"] = fe["load"] - fe["lag_168"]
	fe = fe.dropna(subset=["residual"]).reset_index(drop=True)
	feat_cols = _select_features(fe)
	X = fe[feat_cols]
	y = fe["residual"].values

	OOF = np.full(len(fe), np.nan)
	folds = last_weeks_splits(fe, weeks=3, gap_hours=24)
	fold_scores: List[float] = []
	default = {
		"max_depth": 8,
		"eta": 0.05,
		"subsample": 0.8,
		"colsample_bytree": 0.8,
		"objective": "reg:absoluteerror",
		"n_estimators": 2000,
	}
	if params:
		default.update(params)

	for tr_idx, va_idx in folds:
		tr = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx], feature_names=list(X.columns))
		va = xgb.DMatrix(X.iloc[va_idx], label=y[va_idx], feature_names=list(X.columns))
		watch=[(tr,'train'),(va,'valid')]
		bst = xgb.train({k:v for k,v in default.items() if k!="n_estimators"}, tr, num_boost_round=default["n_estimators"], evals=watch, verbose_eval=False, early_stopping_rounds=200)
		OOF[va_idx] = bst.predict(va) + fe.loc[va_idx, "lag_168"].values
		fold_scores.append(smape(fe.loc[va_idx, "load"].values, np.clip(OOF[va_idx], 0.0, None)))

	mask = ~np.isnan(OOF)
	oof_smape = smape(fe.loc[mask, "load"].values, np.clip(OOF[mask], 0.0, None))
	with open(os.path.join(save_dir, "cv_metrics.json"), "w", encoding="utf-8") as f:
		json.dump({"oof_smape": float(oof_smape), "fold_smape": fold_scores}, f, ensure_ascii=False, indent=2)

	# final
	dtrain = xgb.DMatrix(X, label=y, feature_names=list(X.columns))
	final = xgb.train({k:v for k,v in default.items() if k!="n_estimators"}, dtrain, num_boost_round=default["n_estimators"])
	final.save_model(os.path.join(save_dir, "model.json"))
	with open(os.path.join(save_dir, "feature_names.json"), "w", encoding="utf-8") as f:
		json.dump({"feature_names": feat_cols}, f)
	return {"oof_smape": float(oof_smape)}


def infer_xgb_residual(train_df: pd.DataFrame, test_df: pd.DataFrame, building_info: pd.DataFrame, model_dir: str) -> pd.DataFrame:
	# lag history for test
	_, hist = build_lag_for_test(train_df, test_df)
	hist = hist.rename(columns={"lag168": "lag_168"})
	# features
	use_cols = ["building_id","timestamp","temp","rain","wind","humid","sunshine","irradiance"]
	train_subset = train_df[[c for c in use_cols if c in train_df.columns]].copy(); train_subset["is_test"]=0
	test_subset = test_df[[c for c in use_cols if c in test_df.columns]].copy(); test_subset["is_test"]=1
	combo = pd.concat([train_subset, test_subset], ignore_index=True)
	fb = FeatureBuilder()
	combo_feat, _ = fb.transform(combo, building_info=building_info)
	test_feat = combo_feat[combo_feat["is_test"]==1].copy().reset_index(drop=True)
	test_feat = test_feat.merge(hist[["building_id","timestamp","lag_168","roll_mean_168"]], on=["building_id","timestamp"], how="left")
	with open(os.path.join(model_dir, "feature_names.json"), "r", encoding="utf-8") as f:
		feat_cols = json.load(f)["feature_names"]
	dtest = xgb.DMatrix(test_feat[feat_cols], feature_names=feat_cols)
	model = xgb.Booster()
	model.load_model(os.path.join(model_dir, "model.json"))
	yres = model.predict(dtest)
	yhat = np.clip(test_feat["lag_168"].values + yres, 0.0, None)
	return pd.DataFrame({
		"building_id": test_feat["building_id"].values,
		"timestamp": test_feat["timestamp"].values,
		"pred": yhat,
	})
