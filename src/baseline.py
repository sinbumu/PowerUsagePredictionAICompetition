from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EPS


def build_lag_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	# Ensure sorting for groupby operations
	train_df = train_df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
	test_df = test_df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

	def _lag_roll(df: pd.DataFrame) -> pd.DataFrame:
		g = df.groupby("building_id", sort=False)
		# lag168 from load
		df["lag168"] = g["load"].shift(168)
		# rolling mean over 168 on load with shift(1) to avoid current leakage
		df["roll_mean_168"] = g["load"].shift(1).rolling(168, min_periods=1).mean()
		return df

	train_df = _lag_roll(train_df.copy())
	# For test, we need historical load to compute lag168/rolling; merge last 168+ history from train
	last_hist = train_df[["building_id", "timestamp", "load"]].copy()
	all_df = pd.concat([last_hist, test_df[["building_id", "timestamp"]].assign(load=np.nan)], ignore_index=True)
	all_df = all_df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
	all_df = _lag_roll(all_df)
	# extract only test rows with computed features
	test_feat = all_df.loc[all_df["load"].isna(), ["building_id", "timestamp", "lag168", "roll_mean_168"]].reset_index(drop=True)
	return train_df, test_feat


def predict_baseline(train_df: pd.DataFrame, test_feat: pd.DataFrame) -> pd.Series:
	# simple 0.5/0.5 blend of lag168 and rolling mean, clipped at 0
	yhat = 0.5 * test_feat["lag168"].values + 0.5 * test_feat["roll_mean_168"].values
	yhat = np.clip(yhat, 0.0, None)
	return pd.Series(yhat)
