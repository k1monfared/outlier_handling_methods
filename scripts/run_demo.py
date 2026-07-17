#!/usr/bin/env python3
"""Single entry point. Reproduces the entire demonstration from a fixed seed.

Usage: python scripts/run_demo.py

Steps:
  1. Generate the synthetic metric-reading bank (data/).
  2. Estimate the treatment effect per experiment under every outlier-handling method.
  3. Score each method per metric on production-measurable signals only,
     predictiveness (early reading vs long-term reading) plus a calibration gate
     (early CI covers the eventual estimate), and select the winner. Ground-truth
     recovery is computed for validation only.
  4. Compute decision quality and illustrative business impact.
  5. Write outputs/ (JSON + Markdown) and docs/images/ (figures).
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src import data_gen, decisions, figures, framework, io_utils, reporting


def main():
    t0 = time.time()
    config = io_utils.load_config()
    print(f"[1/5] Generating synthetic bank (seed={config['master_seed']}) ...")
    bank, gt = data_gen.write_bank(config)
    print(f"      experiments={len(gt)}, bank rows={len(bank):,}")

    print("[2/5] Estimating effects under every method for every experiment ...")
    per_exp = framework.compute_per_experiment(config, bank, gt)
    print(f"      per-experiment rows={len(per_exp):,}")

    print("[3/5] Scoring predictiveness with the calibration gate, selecting winners ...")
    summary = framework.summarize(config, per_exp)
    summary, winners = framework.score_and_select(config, summary)
    for metric_name, w in winners.items():
        print(f"      {metric_name}: winner={w['method']} "
              f"(R2={w['pred_r2']:.3f}, early_cover_long={w['early_cover_long']:.3f}"
              f" | validation RMSE={w['rmse_truth']:.4f})")

    print("[4/5] Computing decision quality and illustrative business impact ...")
    dec = decisions.decision_table(config, per_exp)
    impact = decisions.business_impact(config, dec, winners)
    print(f"      mean decision-error reduction vs no handling: "
          f"{impact['program_totals']['mean_decision_error_reduction_pp']:.1f} pp")

    print("[5/5] Writing outputs and figures ...")
    img_paths = figures.generate_all(config, summary, winners, per_exp, dec, bank)
    reporting.write_all(config, summary, winners, per_exp, dec, impact, img_paths)

    print(f"Done in {time.time() - t0:.1f}s. See outputs/ and docs/images/.")


if __name__ == "__main__":
    main()
