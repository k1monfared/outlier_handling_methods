"""Outlier-handling methods.

Every method takes the raw control and treatment arrays for a single reading and
returns handled (control, treatment) arrays. Distribution-based thresholds are
computed on the POOLED sample (control + treatment together) and then applied
identically to both arms. This mirrors production practice and avoids leaking the
treatment assignment into the threshold, which would bias the effect estimate.

Method families covered:
  - no_handling
  - removal / trimming (drop observations beyond a percentile, one or two sided)
  - capping by business rule (fixed constant ceiling)
  - capping by distribution (moment-based: mean + k * sd)
  - winsorization (clip to percentile values, one or two sided)

CUPED is intentionally excluded. CUPED is a variance-reduction technique, not an
outlier-handling method, and is covered by the sibling variance_reduction_methods
repository.
"""
from __future__ import annotations

import numpy as np


def _pooled(c: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.concatenate([c, t])


def apply_method(method: dict, c: np.ndarray, t: np.ndarray, business_cap: float):
    kind = method["kind"]
    if kind == "identity":
        return c, t
    if kind == "trim":
        return _trim(method, c, t)
    if kind == "drop_const":
        return _drop_const(c, t, business_cap)
    if kind == "cap_const":
        return _cap_const(c, t, business_cap)
    if kind == "cap_moment":
        return _cap_moment(method, c, t)
    if kind == "winsor":
        return _winsor(method, c, t)
    raise ValueError(f"unknown method kind: {kind}")


def _trim(method, c, t):
    pool = _pooled(c, t)
    lo = -np.inf
    hi = np.inf
    if "lower_q" in method:
        lo = np.quantile(pool, method["lower_q"])
    if "upper_q" in method:
        hi = np.quantile(pool, method["upper_q"])
    c2 = c[(c >= lo) & (c <= hi)]
    t2 = t[(t >= lo) & (t <= hi)]
    return c2, t2


def _drop_const(c, t, business_cap):
    # Business-rule removal: discard values above the plausible maximum entirely.
    return c[c <= business_cap], t[t <= business_cap]


def _cap_const(c, t, business_cap):
    return np.minimum(c, business_cap), np.minimum(t, business_cap)


def _cap_moment(method, c, t):
    pool = _pooled(c, t)
    hi = pool.mean() + method["k_sd"] * pool.std(ddof=1)
    return np.minimum(c, hi), np.minimum(t, hi)


def _winsor(method, c, t):
    pool = _pooled(c, t)
    lo = -np.inf
    hi = np.inf
    if "lower_q" in method:
        lo = np.quantile(pool, method["lower_q"])
    if "upper_q" in method:
        hi = np.quantile(pool, method["upper_q"])
    return np.clip(c, lo, hi), np.clip(t, lo, hi)
