from __future__ import annotations

import os
from typing import Dict, Optional

import pandas as pd

from .config import COLUMN_MAP, BUILDING_INFO_COLUMN_MAP


def _apply_column_map(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
	rename_cols = {c: mapping[c] for c in df.columns if c in mapping}
	return df.rename(columns=rename_cols)


def load_train_csv(path: str) -> pd.DataFrame:
	df = pd.read_csv(path)
	df = _apply_column_map(df, COLUMN_MAP)
	# parse timestamp, ensure no timezone
	df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d %H", errors="coerce")
	# basic types
	df["building_id"] = df["building_id"].astype(int)
	for col in ["temp", "rain", "wind", "humid", "sunshine", "irradiance", "load"]:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")
	return df


def load_test_csv(path: str) -> pd.DataFrame:
	df = pd.read_csv(path)
	df = _apply_column_map(df, COLUMN_MAP)
	df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d %H", errors="coerce")
	df["building_id"] = df["building_id"].astype(int)
	for col in ["temp", "rain", "wind", "humid"]:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")
	return df


def load_building_info(path: str) -> pd.DataFrame:
	df = pd.read_csv(path)
	df = _apply_column_map(df, BUILDING_INFO_COLUMN_MAP)
	# Normalize dashes to NaN
	df = df.replace({"-": pd.NA})
	df["building_id"] = df["building_id"].astype(int)
	for col in ["total_area", "cooling_area", "pv_capacity", "ess_capacity", "pcs_capacity"]:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors="coerce")
	return df


def align_sample_index(sample_path: str) -> pd.DataFrame:
	idx = pd.read_csv(sample_path)
	return idx
