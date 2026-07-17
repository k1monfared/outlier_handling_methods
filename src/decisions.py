"""Decision quality and illustrative business impact.

The point of better outlier handling is better ship / no-ship decisions. For each
experiment and method we turn the long-term readout into a launch decision and
compare it to the correct decision implied by the known ground-truth effect.

Decision rule (guardrail-style, on the long-term reading):
  - SHIP     if the 95 percent CI lower bound is above 0 (a statistically clear win)
  - NO-SHIP  otherwise

Correct action from ground truth, using a small dead-zone d:
  - true_effect >  d   -> should SHIP        (real win)
  - true_effect < -d   -> should NOT SHIP    (real regression, shipping is costly)
  - otherwise (flat)   -> should NOT SHIP    (no clear benefit)

Two error types are tracked:
  - false_ship : shipped when the truth is flat or a regression (costly wrong launch)
  - missed_win : did not ship a real win (opportunity cost)

The dollar figure is illustrative and derived transparently from stated
assumptions in the config. It is not a real financial result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def decision_table(config: dict, per_exp: pd.DataFrame) -> pd.DataFrame:
    d = config["simulation"]["decision_deadzone"]
    rows = []
    for (metric_name, method_name), g in per_exp.groupby(["metric", "method"], observed=True):
        true = g["true_effect"].to_numpy()
        ci_low = g["long_ci_low"].to_numpy()
        ship = ci_low > 0.0

        should_ship = true > d
        is_regression = true < -d

        false_ship = ship & ~should_ship
        false_ship_regression = ship & is_regression
        missed_win = (~ship) & should_ship

        # Confusion matrix of the ship decision against the ground-truth-correct
        # action (positive = a real win that should ship).
        tp = int((ship & should_ship).sum())      # shipped a real win
        fp = int((ship & ~should_ship).sum())      # shipped a flat or regression (false ship)
        fn = int(((~ship) & should_ship).sum())    # held back a real win (missed win)
        tn = int(((~ship) & ~should_ship).sum())   # correctly did not ship

        n = len(g)
        # Decision regret in relative-effect units: the magnitude of ground-truth
        # effect tied to each wrong decision. A shipped regression costs |true|
        # (a real loss). A missed win costs true (the foregone gain).
        shipped_regression_magnitude = float(np.abs(true[false_ship_regression]).sum())
        missed_win_magnitude = float(true[missed_win].sum())
        regret_magnitude = shipped_regression_magnitude + missed_win_magnitude
        rows.append({
            "metric": metric_name,
            "method": method_name,
            "n_experiments": n,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "false_ship_rate": float(false_ship.mean()),
            "false_ship_regression_rate": float(false_ship_regression.mean()),
            "missed_win_rate": float(missed_win.mean()),
            "decision_error_rate": float((false_ship | missed_win).mean()),
            "decision_accuracy": float((~(false_ship | missed_win)).mean()),
            "shipped_regression_magnitude": shipped_regression_magnitude,
            "missed_win_magnitude": missed_win_magnitude,
            "regret_magnitude": regret_magnitude,
        })
    return pd.DataFrame(rows)


def business_impact(config: dict, dec: pd.DataFrame, winners: dict) -> dict:
    """Compare each metric's winning method to the no-handling baseline and
    translate the improvement into an illustrative annualized figure."""
    bi = config["business_impact"]
    revenue = bi["revenue_at_stake_per_experiment_usd"]
    exp_per_year = bi["experiments_per_year"]

    out = {"assumptions": bi, "per_metric": {}, "program_totals": {}}
    total_avoided = 0.0
    total_err_reduction_pp = []

    for metric_name, w in winners.items():
        win_method = w["method"]
        sub = dec[dec["metric"] == metric_name]
        none_row = sub[sub["method"] == "none"].iloc[0]
        win_row = sub[sub["method"] == win_method].iloc[0]
        n = int(none_row["n_experiments"])

        err_reduction_pp = float(none_row["decision_error_rate"] - win_row["decision_error_rate"]) * 100.0
        total_err_reduction_pp.append(err_reduction_pp)

        # Illustrative cost avoided: total decision regret (shipped regressions
        # plus missed wins) removed by the winner, scaled from the sampled
        # experiments to a yearly program.
        regret_none = float(none_row["regret_magnitude"])
        regret_win = float(win_row["regret_magnitude"])
        regret_avoided_per_sample = max(regret_none - regret_win, 0.0)
        scale = exp_per_year / n if n else 0.0
        avoided_usd = regret_avoided_per_sample * scale * revenue
        total_avoided += avoided_usd

        out["per_metric"][metric_name] = {
            "winning_method": win_method,
            "decision_error_rate_none": float(none_row["decision_error_rate"]),
            "decision_error_rate_winner": float(win_row["decision_error_rate"]),
            "decision_error_reduction_pp": err_reduction_pp,
            "false_ship_rate_none": float(none_row["false_ship_rate"]),
            "false_ship_rate_winner": float(win_row["false_ship_rate"]),
            "confusion_winner": {"tp": int(win_row["tp"]), "fp": int(win_row["fp"]),
                                 "fn": int(win_row["fn"]), "tn": int(win_row["tn"])},
            "confusion_none": {"tp": int(none_row["tp"]), "fp": int(none_row["fp"]),
                               "fn": int(none_row["fn"]), "tn": int(none_row["tn"])},
            "regret_magnitude_none": regret_none,
            "regret_magnitude_winner": regret_win,
            "illustrative_annual_usd_avoided": avoided_usd,
        }

    out["program_totals"] = {
        "mean_decision_error_reduction_pp": float(np.mean(total_err_reduction_pp)),
        "illustrative_total_annual_usd_avoided": float(total_avoided),
        "note": ("Illustrative only. Derived from the simulated decision-error "
                 "reduction and the stated revenue-exposure assumption. Not a real "
                 "financial result."),
    }
    return out
