from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


TIME_FEATURE_COLUMNS = [
	"hour",
	"dayofweek",
	"is_weekend",
	"dayofyear",
	"month",
	"hour_sin",
	"hour_cos",
	"dow_sin",
	"dow_cos",
]

LOAD_FEATURE_COLUMNS = [
	"lag_1",
	"lag_24",
	"lag_168",
	"roll_mean_24",
	"roll_mean_168",
	"roll_std_24",
]

WEATHER_NOW_COLUMNS = ["temp", "rain", "wind", "humid"]
WEATHER_PAST_ONLY_COLUMNS = ["sunshine", "irradiance"]


class FeatureBuilder:
	def __init__(self, use_meta_interactions: bool = True) -> None:
		self.use_meta_interactions = use_meta_interactions

	def fit(self, train_df: pd.DataFrame, building_info: Optional[pd.DataFrame] = None) -> "FeatureBuilder":
		# Currently stateless; placeholder for future scalers/encoders.
		return self

	def transform(
		self,
		df: pd.DataFrame,
		building_info: Optional[pd.DataFrame] = None,
		is_test: bool = False,
	) -> Tuple[pd.DataFrame, List[str]]:
		"""Create leak-safe features.

		- Time features
		- Load lags/rolling (requires 'load' for meaningful values)
		- Weather now (temp/rain/wind/humid) and lag168; diffs for now-vars only
		- Sunshine/Irradiance: past-only (lag/rolling) when columns exist
		- Meta interactions with PV capacity when available
		"""
		df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

		# Merge building info if provided
		if building_info is not None:
			if "building_id" in building_info.columns:
				df = df.merge(building_info, on="building_id", how="left")

		feature_cols: List[str] = []

		# Time features
		df["hour"] = df["timestamp"].dt.hour
		df["dayofweek"] = df["timestamp"].dt.dayofweek
		df["is_weekend"] = (df["dayofweek"].isin([5, 6])).astype(int)
		df["dayofyear"] = df["timestamp"].dt.dayofyear
		df["month"] = df["timestamp"].dt.month
		# Cyclical encodings
		df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
		df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
		df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
		df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
		feature_cols += TIME_FEATURE_COLUMNS

		# Load-based features (if load exists)
		if "load" in df.columns:
			g = df.groupby("building_id", sort=False)
			df["lag_1"] = g["load"].shift(1)
			df["lag_24"] = g["load"].shift(24)
			df["lag_168"] = g["load"].shift(168)
			# rolling with shift(1) to avoid leakage
			df["roll_mean_24"] = g["load"].shift(1).rolling(24, min_periods=1).mean()
			df["roll_mean_168"] = g["load"].shift(1).rolling(168, min_periods=1).mean()
			df["roll_std_24"] = g["load"].shift(1).rolling(24, min_periods=2).std()
			feature_cols += LOAD_FEATURE_COLUMNS

		# Weather now (available in both train/test for these variables)
		g = df.groupby("building_id", sort=False)
		for col in WEATHER_NOW_COLUMNS:
			if col in df.columns:
				lag_col = f"{col}_lag168"
				df[lag_col] = g[col].shift(168)
				feature_cols.append(lag_col)
				# diff: now - lag168
				diff_col = f"{col}_diff168"
				df[diff_col] = df[col] - df[lag_col]
				feature_cols.append(diff_col)

		# Sunshine/Irradiance: past-only when present in df
		for col in WEATHER_PAST_ONLY_COLUMNS:
			if col in df.columns:
				lag_col = f"{col}_lag168"
				df[lag_col] = g[col].shift(168)
				feature_cols.append(lag_col)
				roll_col = f"{col}_roll_mean_168"
				df[roll_col] = g[col].shift(1).rolling(168, min_periods=1).mean()
				feature_cols.append(roll_col)

		# Cooling/Heating degree days from temp
		if "temp" in df.columns:
			df["cdd"] = (df["temp"] - 24.0).clip(lower=0.0)
			df["hdd"] = (18.0 - df["temp"]).clip(lower=0.0)
			feature_cols += ["cdd", "hdd"]

		# Meta interactions (PV × irradiance lag/rolling)
		if self.use_meta_interactions and ("pv_capacity" in df.columns):
			for base in ["irradiance_lag168", "irradiance_roll_mean_168", "sunshine_lag168", "sunshine_roll_mean_168"]:
				if base in df.columns:
					name = f"pvx_{base}"
					df[name] = df["pv_capacity"] * df[base]
					feature_cols.append(name)

		return df, feature_cols
