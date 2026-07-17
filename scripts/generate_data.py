#!/usr/bin/env python3
"""Generate and commit the synthetic metric-reading bank.

Usage: python scripts/generate_data.py
Writes data/metric_bank.parquet, data/ground_truth.csv, data/metric_bank_sample.csv.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data_gen, io_utils


def main():
    config = io_utils.load_config()
    print(f"Generating synthetic bank (seed={config['master_seed']}) ...")
    bank, gt = data_gen.write_bank(config)
    n_metrics = gt["metric"].nunique()
    n_exp = len(gt)
    print(f"  metrics: {n_metrics}, experiments: {n_exp}, bank rows: {len(bank):,}")
    print(f"  wrote {io_utils.BANK_PARQUET}")
    print(f"  wrote {io_utils.GROUND_TRUTH_CSV}")
    print(f"  wrote {io_utils.SAMPLE_CSV}")


if __name__ == "__main__":
    main()
