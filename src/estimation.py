"""Treatment-effect estimation for a single reading.

We estimate the RELATIVE treatment effect on the mean,

    rel = (mean_treatment - mean_control) / mean_control,

and its 95 percent confidence interval via the delta method. This matches how a
percent-change guardrail readout is reported on an experimentation platform. The
same estimator is applied after every outlier-handling method, so differences in
bias, variance, and coverage come purely from the handling.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Z95 = 1.959963984540054


@dataclass
class EffectEstimate:
    rel: float          # relative effect estimate
    se: float           # standard error of the relative effect (delta method)
    ci_low: float
    ci_high: float
    n_c: int
    n_t: int
    mean_c: float
    mean_t: float


def estimate_relative_effect(c: np.ndarray, t: np.ndarray) -> EffectEstimate:
    n_c = c.shape[0]
    n_t = t.shape[0]
    mean_c = float(c.mean())
    mean_t = float(t.mean())
    var_c = float(c.var(ddof=1))
    var_t = float(t.var(ddof=1))

    rel = (mean_t - mean_c) / mean_c

    # Delta method for rel = mean_t / mean_c - 1, treating the two arm means as
    # independent: Var(rel) approx Var(mean_t)/mean_c^2 + mean_t^2 * Var(mean_c)/mean_c^4.
    v_mt = var_t / n_t
    v_mc = var_c / n_c
    var_rel = v_mt / (mean_c ** 2) + (mean_t ** 2) * v_mc / (mean_c ** 4)
    se = math.sqrt(var_rel) if var_rel > 0 else 0.0

    return EffectEstimate(
        rel=rel, se=se,
        ci_low=rel - Z95 * se, ci_high=rel + Z95 * se,
        n_c=n_c, n_t=n_t, mean_c=mean_c, mean_t=mean_t,
    )
