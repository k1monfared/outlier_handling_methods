#!/usr/bin/env python3
"""Regenerate figures from the committed data and a fresh analysis run.

Usage: python scripts/generate_figures.py
Normally you do not need to call this directly. scripts/run_demo.py runs it.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src import decisions, figures, framework, io_utils, data_gen


def main():
    config = io_utils.load_config()
    if not os.path.exists(io_utils.BANK_PARQUET):
        data_gen.write_bank(config)
    bank = pd.read_parquet(io_utils.BANK_PARQUET)
    gt = pd.read_csv(io_utils.GROUND_TRUTH_CSV)

    per_exp = framework.compute_per_experiment(config, bank, gt)
    summary = framework.summarize(config, per_exp)
    summary, winners = framework.score_and_select(config, summary)
    dec = decisions.decision_table(config, per_exp)

    paths = figures.generate_all(config, summary, winners, per_exp, dec, bank)
    for p in paths:
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
