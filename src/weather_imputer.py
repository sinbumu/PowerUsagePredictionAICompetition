from __future__ import annotations

import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load

from .cv import last_weeks_splits

try:
	import lightgbm as lgb
	_HAS_LGBM = True
except Exception:
	_HAS_LGBM = False

from sklearn.ensemble import HistGradientBoostingRegressor


FEATURES = ["hour", "dayofweek", "dayofyear", "temp", "humid", "wind", "rain"]
TARGETS = ["sunshine", "irradiance"]


def _build_imputer_frame(df: pd.DataFrame) -> pd.DataFrame:
	f = df.copy()
	f = f.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
	f["hour"] = f["timestamp"].dt.hour
	f["dayofweek"] = f["timestamp"].dt.dayofweek
	f["dayofyear"] = f["timestamp"].dt.dayofyear
	return f


def generate_now_hat(
	train_df: pd.DataFrame,
	test_df: pd.DataFrame,
	save_dir: str = "outputs/models/weather_imputer",
	use_last_weeks_cv: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	os.makedirs(save_dir, exist_ok=True)
	train_f = _build_imputer_frame(train_df)
	test_f = _build_imputer_frame(test_df)

	train_oof: Dict[str, np.ndarray] = {}
	test_pred: Dict[str, np.ndarray] = {}

	for tgt in TARGETS:
		# fit per target
		cols = [c for c in FEATURES if c in train_f.columns]
		X = train_f[cols].copy()
		y = train_f[tgt].astype(float).values
		Xt = test_f[cols].copy()

		folds = last_weeks_splits(train_f, weeks=3, gap_hours=24) if use_last_weeks_cv else []
		OOF = np.full(len(train_f), np.nan)
		models = []
		for tr_idx, va_idx in folds:
			if _HAS_LGBM:
				m = lgb.LGBMRegressor(objective="mae", n_estimators=2000, learning_rate=0.05)
				m.fit(X.iloc[tr_idx], y[tr_idx], eval_set=[(X.iloc[va_idx], y[va_idx])], eval_metric="mae", callbacks=[lgb.early_stopping(200, verbose=False)])
			else:
				m = HistGradientBoostingRegressor(max_iter=800, learning_rate=0.05)
				m.fit(X.iloc[tr_idx], y[tr_idx])
			OOF[va_idx] = m.predict(X.iloc[va_idx])
			models.append(m)
		# final fit
		if _HAS_LGBM:
			final = lgb.LGBMRegressor(objective="mae", n_estimators= int(np.mean([getattr(m, 'best_iteration_', 2000) or 2000 for m in models]) or 1000), learning_rate=0.05)
			final.fit(X, y)
		else:
			final = HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05)
			final.fit(X, y)
		dump(final, os.path.join(save_dir, f"{tgt}.pkl"))
		train_oof[tgt] = OOF
		test_pred[tgt] = final.predict(Xt)

	# attach to copies
	train_aug = train_df.copy().reset_index(drop=True)
	test_aug = test_df.copy().reset_index(drop=True)
	for tgt in TARGETS:
		train_aug[f"{tgt}_now_hat_oof"] = train_oof[tgt]
		test_aug[f"{tgt}_now_hat"] = test_pred[tgt]
	# save summary
	with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
		json.dump({"features": [c for c in FEATURES if c in train_f.columns], "targets": TARGETS}, f)
	return train_aug, test_aug
