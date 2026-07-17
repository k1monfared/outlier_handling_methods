"""The selection framework.

Selection uses only signals that are measurable in production, where the true
effect of a live experiment is never known. The reference is the long-term
reading, the eventual estimate you would trust after letting an experiment run,
learned across many long-running experiments. For each metric and each candidate
outlier-handling method we compute across all simulated experiments:

  (1) PREDICTIVENESS (the selection score). How well the early short-window
      reading predicts the long-term reading of the same experiment. We report
      the Pearson correlation, R squared, the slope, and the RMSE between early
      and long readings across experiments.

  (2) EARLY-CI CALIBRATION (the validity gate). How often the early 95 percent CI
      covers the eventual long-term estimate. A method whose early CI
      systematically misses where the experiment lands is over-confident, usually
      from over-aggressive trimming, and is disqualified from winning.

Both are computed without ever looking at the injected truth. The best eligible
method per metric is the one with the highest predictiveness, with a simplicity
tiebreak.

Ground-truth recovery (bias, RMSE, and CI coverage against the KNOWN true effect)
is still computed, but ONLY to validate the choice after the fact: it shows
whether the method selected on production-measurable signals actually recovers the
true effect. It never enters selection, because it cannot be computed on a live
experiment. The point of the framework is that the winner differs across metrics,
and the production-measurable numbers explain why.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import methods as methods_mod
from .estimation import estimate_relative_effect


def _reading_arrays(bank: pd.DataFrame):
    """Index the bank for fast lookup: (metric, exp_id, reading, arm) -> ndarray."""
    idx = {}
    grouped = bank.groupby(["metric", "exp_id", "reading", "arm"], observed=True)["value"]
    for key, series in grouped:
        idx[key] = series.to_numpy(dtype=np.float64)
    return idx


def compute_per_experiment(config: dict, bank: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    """One row per (metric, method, exp_id) with early and long estimates."""
    idx = _reading_arrays(bank)
    metrics = config["metrics"]
    methods = config["methods"]
    gt_lookup = {(r.metric, r.exp_id): r.true_effect for r in gt.itertuples()}

    rows = []
    for metric_name, mcfg in metrics.items():
        business_cap = mcfg["business_cap"]
        exp_ids = sorted(gt[gt["metric"] == metric_name]["exp_id"].unique())
        for method in methods:
            for e in exp_ids:
                true_effect = gt_lookup[(metric_name, e)]
                rec = {
                    "metric": metric_name,
                    "method": method["name"],
                    "exp_id": e,
                    "true_effect": true_effect,
                }
                for reading in ("long", "early"):
                    c = idx[(metric_name, e, reading, "control")]
                    t = idx[(metric_name, e, reading, "treatment")]
                    hc, ht = methods_mod.apply_method(method, c, t, business_cap)
                    est = estimate_relative_effect(hc, ht)
                    rec[f"{reading}_rel"] = est.rel
                    rec[f"{reading}_se"] = est.se
                    rec[f"{reading}_ci_low"] = est.ci_low
                    rec[f"{reading}_ci_high"] = est.ci_high
                rows.append(rec)
    return pd.DataFrame(rows)


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def summarize(config: dict, per_exp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-experiment rows to one row per (metric, method) with the
    accuracy and predictiveness diagnostics."""
    rows = []
    for (metric_name, method_name), g in per_exp.groupby(["metric", "method"], observed=True):
        true = g["true_effect"].to_numpy()
        long_rel = g["long_rel"].to_numpy()
        early_rel = g["early_rel"].to_numpy()

        # Validation only (never used to select): recovery of the known truth.
        err = long_rel - true
        bias = float(err.mean())
        variance = float(long_rel.var(ddof=1))
        rmse_truth = _rmse(long_rel, true)
        mse_truth = rmse_truth ** 2
        covered = (g["long_ci_low"].to_numpy() <= true) & (true <= g["long_ci_high"].to_numpy())
        coverage = float(covered.mean())
        mean_ci_width = float((g["long_ci_high"] - g["long_ci_low"]).mean())

        # (2) Validity gate, production-measurable: does the early 95 percent CI
        # cover the eventual long-term estimate. No ground truth involved.
        early_lo = g["early_ci_low"].to_numpy()
        early_hi = g["early_ci_high"].to_numpy()
        early_cover_long = float(((early_lo <= long_rel) & (long_rel <= early_hi)).mean())

        # (1) Predictiveness: early reading predicting long reading.
        pred_rmse = _rmse(early_rel, long_rel)
        if np.std(early_rel) > 0 and np.std(long_rel) > 0:
            pearson = float(np.corrcoef(early_rel, long_rel)[0, 1])
        else:
            pearson = float("nan")
        r2 = pearson ** 2 if pearson == pearson else float("nan")
        # Slope of long on early (how faithfully early tracks long).
        if np.var(early_rel) > 0:
            slope = float(np.cov(early_rel, long_rel, ddof=1)[0, 1] / np.var(early_rel, ddof=1))
        else:
            slope = float("nan")
        early_bias = float((early_rel - true).mean())

        rows.append({
            "metric": metric_name,
            "method": method_name,
            "bias": bias,
            "variance": variance,
            "rmse_truth": rmse_truth,
            "mse_truth": mse_truth,
            "ci_coverage": coverage,
            "mean_ci_width": mean_ci_width,
            "early_cover_long": early_cover_long,
            "early_bias": early_bias,
            "pred_rmse": pred_rmse,
            "pred_pearson": pearson,
            "pred_r2": r2,
            "pred_slope": slope,
        })
    return pd.DataFrame(rows)


def score_and_select(config: dict, summary: pd.DataFrame):
    """Apply the calibration gate, score eligible methods, and select the winner.

    Stage 1 (calibration gate): a method is eligible to win only if its early 95
    percent CI covers the eventual long-term estimate at least calibration_gate of
    the time. A method whose early interval systematically misses where the
    experiment lands is over-confident (usually from over-aggressive trimming), so
    it is disqualified. This gate is measurable in production, because it compares
    the early CI to the eventual reading, never to the unknown true effect.

    Stage 2 (scoring): the score is predictiveness, measured as the early-to-long
    R squared (how much of the long-term reading's variation across experiments the
    early reading explains) and ratio-normalized to the best R squared among
    ELIGIBLE methods within the metric, so the best eligible method scores 1.0. We
    deliberately score R squared, not the early-to-long RMSE, because RMSE is gamed
    by aggressive shrinkage: a method that clips almost everything drives both the
    early and long readings toward the same constant, making their RMSE tiny while
    destroying any real tracking. R squared collapses under that over-clipping, so
    it rewards genuine predictiveness. Within tiebreak_epsilon of the best score,
    the simplest method is preferred.

    Ground-truth recovery is never used here. It is reported alongside for
    validation only.
    """
    scoring = config["scoring"]
    gate = scoring["calibration_gate"]
    eps = scoring["tiebreak_epsilon"]
    order = {name: i for i, name in enumerate(scoring["simplicity_order"])}

    summary = summary.copy()
    summary["eligible"] = summary["early_cover_long"] >= gate
    summary["predictiveness_score"] = np.nan
    summary["composite_score"] = np.nan

    winners = {}
    for metric_name, g in summary.groupby("metric", observed=True):
        eligible = g[g["eligible"]]
        # If nothing passes the gate, fall back to the best-calibrated method(s).
        pool = eligible if len(eligible) else g[g["early_cover_long"] == g["early_cover_long"].max()]

        best_pred = pool["pred_r2"].max()
        if best_pred > 0:
            pred = g["pred_r2"] / best_pred
        else:
            pred = g["pred_r2"] * 0.0
        # The composite is predictiveness alone; kept under this name so the
        # reports and explorer that read composite_score need no rewiring.
        summary.loc[g.index, "predictiveness_score"] = pred.to_numpy()
        summary.loc[g.index, "composite_score"] = pred.to_numpy()

        # Winner: highest predictiveness among eligible, simplicity tiebreak inside epsilon.
        gg = summary.loc[pool.index].copy()
        top = gg["composite_score"].max()
        contenders = gg[gg["composite_score"] >= top - eps]
        contenders = contenders.assign(_simp=contenders["method"].map(
            lambda m: order.get(m, 999)))
        winner_row = contenders.sort_values(
            ["_simp", "composite_score"], ascending=[True, False]).iloc[0]
        winners[metric_name] = {
            "method": winner_row["method"],
            "composite_score": float(winner_row["composite_score"]),
            "pred_r2": float(winner_row["pred_r2"]),
            "pred_rmse": float(winner_row["pred_rmse"]),
            "early_cover_long": float(winner_row["early_cover_long"]),
            # validation only, not used in selection
            "rmse_truth": float(winner_row["rmse_truth"]),
            "bias": float(winner_row["bias"]),
            "variance": float(winner_row["variance"]),
            "ci_coverage": float(winner_row["ci_coverage"]),
            "n_eligible": int(len(eligible)),
            "n_contenders_within_eps": int(len(contenders)),
            "disqualified_by_gate": sorted(g[~g["eligible"]]["method"].tolist()),
        }

    return summary.sort_values(["metric", "composite_score"],
                               ascending=[True, False]).reset_index(drop=True), winners
