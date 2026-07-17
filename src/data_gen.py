"""Synthetic metric-reading bank generator.

Builds a bank of simulated A/B experiments across three metric archetypes.
Each experiment has a KNOWN ground-truth relative treatment effect (defined on
the clean data-generating process) and produces two readings:

  - long  : the long-term reading (larger sample, effect fully realized)
  - early : an early short-window reading (smaller sample, same true effect,
            noisier because fewer observations)

For the two heavy-tailed metrics we inject contamination that is present in both
arms and is NOT caused by the treatment (a logging bug for revenue, bots for
engagement). The genuine tail (real high spenders, real power users) is left in
place, so the "outliers are sometimes the real signal" point is real: some of
the tail must be kept, some must be removed, and which is which depends on the
metric.

All randomness derives from the master seed via numpy SeedSequence spawning, so
the bank is byte-for-byte reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import io_utils


def _draw_true_effects(rng: np.random.Generator, sim: dict, n: int) -> np.ndarray:
    """Per-experiment ground-truth relative effect on the clean metric mean."""
    effects = rng.normal(sim["true_effect_mean"], sim["true_effect_sd"], size=n)
    effects = np.clip(effects, sim["true_effect_clip_low"], sim["true_effect_clip_high"])
    # A fraction are exact nulls (A/A-like), useful for coverage and false-ship checks.
    is_null = rng.random(n) < sim["frac_null_experiments"]
    effects[is_null] = 0.0
    return effects


def _gen_revenue(rng, mcfg, delta, n):
    """Heavy-tailed revenue with the treatment effect living in the tail.

    Each user is either a small spender or a whale. The treatment works by
    converting more users into whales (a whale-frequency effect), so the entire
    relative mean effect sits in the tail. The whale probability under treatment
    is solved analytically so the clean mean changes by exactly delta. Impossible
    contamination (a logging bug) is then injected identically into both arms.

    This is why a percentile-based cut fails here: clipping the tail removes the
    genuine whales that carry the effect. Only removing the impossible values by
    business rule keeps the signal while deleting the contamination.
    """
    b = mcfg["base"]
    scale = b["scale"]
    mean_small = scale * np.exp(b["small_meanlog"] + 0.5 * b["small_sdlog"] ** 2)
    mean_whale = scale * np.exp(b["whale_meanlog"] + 0.5 * b["whale_sdlog"] ** 2)
    p_c = b["p_whale_control"]
    mean_c = (1.0 - p_c) * mean_small + p_c * mean_whale
    # Solve treatment whale probability so the clean mean shifts by delta.
    target_mean_t = mean_c * (1.0 + delta)
    p_t = (target_mean_t - mean_small) / (mean_whale - mean_small)
    p_t = float(np.clip(p_t, 0.0, 1.0))

    def draw(size, p_whale):
        is_whale = rng.random(size) < p_whale
        small = rng.lognormal(b["small_meanlog"], b["small_sdlog"], size=size) * scale
        whale = rng.lognormal(b["whale_meanlog"], b["whale_sdlog"], size=size) * scale
        return np.where(is_whale, whale, small)

    clean_c = draw(n, p_c)
    clean_t = draw(n, p_t)
    c = _inject_impossible(rng, clean_c, mcfg["contamination"])
    t = _inject_impossible(rng, clean_t, mcfg["contamination"])
    return c, t


def _gen_engagement(rng, mcfg, delta, n):
    """Heavy-tailed counts. Gamma-Poisson humans with a genuine power-user tail,
    multiplicative effect on the human rate, plus bot-flood contamination in both
    arms (bots are unaffected by the treatment)."""
    b = mcfg["base"]

    def draw_rates(size):
        rate = rng.gamma(b["human_shape"], b["human_scale"], size=size)
        is_power = rng.random(size) < b["power_user_frac"]
        n_power = int(is_power.sum())
        if n_power:
            rate[is_power] = rng.gamma(b["power_shape"], b["power_scale"], size=n_power)
        return rate

    rate_c = draw_rates(n)
    rate_t = draw_rates(n) * (1.0 + delta)
    c = rng.poisson(rate_c).astype(np.float64)
    t = rng.poisson(rate_t).astype(np.float64)
    c = _inject_bots(rng, c, mcfg["contamination"])
    t = _inject_bots(rng, t, mcfg["contamination"])
    return c, t


def _gen_conversion(rng, mcfg, delta, n):
    """Bounded per-user conversion rate = successes / visits. Multiplicative
    effect on the per-user conversion probability. No contamination: extreme
    0.0 and 1.0 values are legitimate low-visit users."""
    b = mcfg["base"]

    def draw(size, mult):
        visits = b["visits_min"] + rng.poisson(b["visits_extra_poisson"], size=size)
        p = rng.beta(b["p_alpha"], b["p_beta"], size=size)
        p = np.clip(p * mult, 0.0, 1.0)
        succ = rng.binomial(visits, p)
        return succ / visits

    c = draw(n, 1.0)
    t = draw(n, 1.0 + delta)
    return c, t


def _inject_impossible(rng, arr, cfg):
    if cfg["rate"] <= 0:
        return arr
    out = arr.copy()
    mask = rng.random(arr.shape[0]) < cfg["rate"]
    k = int(mask.sum())
    if k:
        out[mask] = rng.uniform(cfg["low"], cfg["high"], size=k)
    return out


def _inject_bots(rng, arr, cfg):
    if cfg["rate"] <= 0:
        return arr
    out = arr.copy()
    mask = rng.random(arr.shape[0]) < cfg["rate"]
    k = int(mask.sum())
    if k:
        out[mask] = np.floor(rng.uniform(cfg["low"], cfg["high"], size=k))
    return out


_GENERATORS = {
    "revenue_per_user": _gen_revenue,
    "engagement_events_per_user": _gen_engagement,
    "conversion_rate_per_user": _gen_conversion,
}


def generate_bank(config: dict):
    """Return (bank_df, ground_truth_df).

    bank_df is long format: metric, exp_id, reading, arm, value (one row per
    simulated user reading). ground_truth_df is one row per experiment.
    """
    sim = config["simulation"]
    metrics = config["metrics"]
    n_exp = sim["n_experiments_per_metric"]
    n_long = sim["n_users_long_per_arm"]
    n_early = sim["n_users_early_per_arm"]

    root_ss = np.random.SeedSequence(config["master_seed"])
    # One independent SeedSequence per metric, then per experiment, then per reading.
    metric_ss = dict(zip(metrics.keys(), root_ss.spawn(len(metrics))))

    bank_frames = []
    gt_rows = []

    for metric_name, mcfg in metrics.items():
        gen = _GENERATORS[metric_name]
        exp_ss = metric_ss[metric_name].spawn(n_exp)
        # Draw all true effects for this metric from a dedicated stream.
        eff_rng = np.random.default_rng(metric_ss[metric_name].spawn(1)[0])
        true_effects = _draw_true_effects(eff_rng, sim, n_exp)

        for e in range(n_exp):
            delta = float(true_effects[e])
            reading_ss = exp_ss[e].spawn(2)
            for reading, ss, n in (("long", reading_ss[0], n_long),
                                   ("early", reading_ss[1], n_early)):
                rng = np.random.default_rng(ss)
                c, t = gen(rng, mcfg, delta, n)
                for arm, vals in (("control", c), ("treatment", t)):
                    bank_frames.append(pd.DataFrame({
                        "metric": metric_name,
                        "exp_id": e,
                        "reading": reading,
                        "arm": arm,
                        "value": vals.astype(np.float32),
                    }))

            gt_rows.append({
                "metric": metric_name,
                "exp_id": e,
                "true_effect": delta,
                "is_null": delta == 0.0,
                "n_users_long_per_arm": n_long,
                "n_users_early_per_arm": n_early,
            })

    bank_df = pd.concat(bank_frames, ignore_index=True)
    for col in ("metric", "reading", "arm"):
        bank_df[col] = bank_df[col].astype("category")
    gt_df = pd.DataFrame(gt_rows)
    return bank_df, gt_df


def write_bank(config: dict):
    """Generate and persist the bank, ground truth, and a small inspectable sample."""
    io_utils.ensure_dirs()
    bank_df, gt_df = generate_bank(config)
    bank_df.to_parquet(io_utils.BANK_PARQUET, index=False)
    gt_df.to_csv(io_utils.GROUND_TRUTH_CSV, index=False)

    # A tiny plain-CSV sample (experiment 0 of each metric, both readings) so the
    # bank is inspectable without a parquet reader.
    sample = bank_df[bank_df["exp_id"] == 0].copy()
    sample.to_csv(io_utils.SAMPLE_CSV, index=False)

    return bank_df, gt_df
