#!/usr/bin/env python3
from __future__ import annotations

import sys
import pandas as pd


def check_submission(path_sub: str, path_sample: str) -> None:
	sub = pd.read_csv(path_sub)
	sample = pd.read_csv(path_sample)
	assert list(sub.columns) == ["num_date_time", "answer"], "submission columns mismatch"
	assert len(sub) == len(sample), "row count mismatch"
	assert sub["num_date_time"].equals(sample["num_date_time"]), "ID order mismatch"
	assert sub["answer"].notna().all(), "NaN in answers"
	assert (sub["answer"] >= 0).all(), "negative values in answers"
	print("submission check: OK")


def main():
	if len(sys.argv) != 3:
		print("Usage: check_quality.py <submission.csv> <sample_submission.csv>")
		sys.exit(1)
	check_submission(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
	main()
