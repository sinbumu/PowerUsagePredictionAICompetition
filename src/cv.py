from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass
class BlockTimeSeriesSplit:
	block_hours: int = 168  # 7 days
	gap_hours: int = 24
	min_train_blocks: int = 1

	def split(self, df: pd.DataFrame, timestamp_col: str = "timestamp") -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
		"""Yield (train_idx, valid_idx) for consecutive 7-day validation blocks with a gap before each valid block.
		Assumes df is sorted by timestamp per group already; We split globally by timestamp.
		"""
		if not np.issubdtype(df[timestamp_col].dtype, np.datetime64):
			raise ValueError("timestamp column must be datetime64[ns]")

		# Determine time range
		start = df[timestamp_col].min()
		end = df[timestamp_col].max()

		block = pd.Timedelta(hours=self.block_hours)
		gap = pd.Timedelta(hours=self.gap_hours)

		current_valid_start = start + block * self.min_train_blocks
		while current_valid_start + block <= end + pd.Timedelta(hours=1):
			valid_start = current_valid_start
			valid_end = valid_start + block
			train_end = valid_start - gap

			train_idx = df.index[df[timestamp_col] < train_end].to_numpy()
			valid_idx = df.index[(df[timestamp_col] >= valid_start) & (df[timestamp_col] < valid_end)].to_numpy()

			if len(train_idx) > 0 and len(valid_idx) > 0:
				yield train_idx, valid_idx

			current_valid_start = valid_start + block
