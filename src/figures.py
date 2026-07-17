"""Figure generation. All figures are produced from the actual run results.

Uses matplotlib only (Agg backend), a neutral palette, and no emojis.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import io_utils

# Neutral, colorblind-friendly palette.
INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
BASE = "#9aa5b1"
WINNER = "#2f7d5b"
ACCENT = "#3a6ea5"
WARN = "#b45309"
CONTAM = "#b23a48"

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

METRIC_TITLES = {
    "revenue_per_user": "Revenue per user (heavy-tailed: signal AND contamination in the tail)",
    "engagement_events_per_user": "Engagement events per user (heavy-tailed: bot contamination in the tail)",
    "conversion_rate_per_user": "Conversion rate per user (bounded: no real outliers)",
}


def _method_order(config):
    return [m["name"] for m in config["methods"]]


def _bar_colors(methods, winner):
    return [WINNER if m == winner else (WARN if m == "none" else BASE) for m in methods]


def per_metric_comparison(config, summary, winners):
    paths = []
    order = _method_order(config)
    for i, metric_name in enumerate(config["metrics"].keys(), start=1):
        g = summary[summary["metric"] == metric_name].set_index("method").reindex(order)
        winner = winners[metric_name]["method"]
        colors = _bar_colors(order, winner)
        # Hatch bars for methods disqualified by the validity gate.
        eligible = g["eligible"].to_numpy() if "eligible" in g else np.ones(len(order), bool)
        hatches = ["" if e else "xxx" for e in eligible]

        fig, axes = plt.subplots(1, 3, figsize=(20, 5.4))
        fig.suptitle(METRIC_TITLES[metric_name], fontsize=12, fontweight="bold")

        def _bars(ax, values):
            bars = ax.bar(order, values, color=colors)
            for b, h in zip(bars, hatches):
                if h:
                    b.set_hatch(h)
                    b.set_edgecolor(CONTAM)

        gate = config["scoring"]["calibration_gate"]

        _bars(axes[0], g["pred_r2"].to_numpy())
        axes[0].set_title("Predictiveness (selection score):\nearly vs long-term R squared (higher is better)")
        axes[0].set_ylabel("R squared (early predicts long)")
        axes[0].set_ylim(0, 1)

        _bars(axes[1], g["early_cover_long"].to_numpy())
        axes[1].axhline(gate, color=INK, linewidth=1, linestyle="--")
        axes[1].set_title("Calibration gate:\nearly CI covers the eventual estimate (above dashed line is eligible)")
        axes[1].set_ylabel("share of experiments covered")
        axes[1].set_ylim(0, 1)

        _bars(axes[2], g["rmse_truth"].to_numpy())
        axes[2].set_title("Validation only: RMSE vs ground truth\n(not used to select, lower is better)")
        axes[2].set_ylabel("RMSE of relative effect")

        for ax in axes:
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=90, ha="center", fontsize=6.5)

        handles = [
            plt.Rectangle((0, 0), 1, 1, color=WINNER),
            plt.Rectangle((0, 0), 1, 1, color=WARN),
            plt.Rectangle((0, 0), 1, 1, color=BASE),
            plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=CONTAM, hatch="xxx"),
        ]
        fig.legend(handles, ["selected winner", "no handling (baseline)", "other methods",
                             "disqualified by calibration gate (over-confident)"],
                   loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.04, 1, 0.95])
        path = os.path.join(io_utils.IMAGES_DIR, f"m{i}_{metric_name}_comparison.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def composite_heatmap(config, summary, winners):
    order = _method_order(config)
    metric_names = list(config["metrics"].keys())
    mat = np.full((len(metric_names), len(order)), np.nan)
    for r, m in enumerate(metric_names):
        g = summary[summary["metric"] == m].set_index("method")
        for c, meth in enumerate(order):
            mat[r, c] = g.loc[meth, "composite_score"]

    # Eligibility per (metric, method) for marking disqualified cells.
    elig = {}
    for r, m in enumerate(metric_names):
        gg = summary[summary["metric"] == m].set_index("method")
        for meth in order:
            elig[(r, meth)] = bool(gg.loc[meth, "eligible"]) if "eligible" in gg else True

    # Color by composite clipped at 1.0 so a disqualified, high-scoring cell does
    # not out-shine the selected winner.
    color_mat = np.clip(mat, None, 1.0)

    fig, ax = plt.subplots(figsize=(20, 4.8))
    im = ax.imshow(color_mat, aspect="auto", cmap="YlGn", vmin=np.nanmin(mat), vmax=1.0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, ha="center", fontsize=7)
    ax.set_yticks(range(len(metric_names)))
    ax.set_yticklabels([m.replace("_", " ") for m in metric_names], fontsize=9)
    ax.set_title("Predictiveness score by metric and method (1.0 = best eligible in row). "
                 "Selected winner outlined; X = disqualified by calibration gate.",
                 fontsize=11, fontweight="bold")
    for r in range(len(metric_names)):
        for c, meth in enumerate(order):
            val = mat[r, c]
            disq = not elig[(r, meth)]
            ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color=INK if color_mat[r, c] > 0.6 else "white")
            if disq:
                ax.plot([c - 0.45, c + 0.45], [r - 0.45, r + 0.45],
                        color=CONTAM, linewidth=1.6)
                ax.plot([c - 0.45, c + 0.45], [r + 0.45, r - 0.45],
                        color=CONTAM, linewidth=1.6)
        winner = winners[metric_names[r]]["method"]
        wc = order.index(winner)
        ax.add_patch(plt.Rectangle((wc - 0.5, r - 0.5), 1, 1, fill=False,
                                   edgecolor=ACCENT, linewidth=2.8))
    fig.colorbar(im, ax=ax, label="predictiveness score", shrink=0.8)
    ax.grid(False)
    fig.tight_layout()
    path = os.path.join(io_utils.IMAGES_DIR, "composite_heatmap.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def revenue_signal_illustration(config, bank):
    """Show, for one revenue experiment, that the tail holds BOTH genuine spenders
    (the signal) and impossible contamination, and where aggressive vs light
    winsorization land. Aggressive (95th) cuts into the genuine spender bump and
    is disqualified for bias; light (99th) sits just above it and removes only the
    extreme contamination, which is why it wins for this metric."""
    sub = bank[(bank["metric"] == "revenue_per_user")
               & (bank["exp_id"] == 0) & (bank["reading"] == "long")]
    vals = sub["value"].to_numpy(dtype=float)
    contam_low = config["metrics"]["revenue_per_user"]["contamination"]["low"]
    genuine = vals[vals < contam_low]
    contam = vals[vals >= contam_low]

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.logspace(0, np.log10(vals.max() * 1.1), 60)
    ax.hist(genuine, bins=bins, color=ACCENT, alpha=0.8,
            label="genuine users, incl. real high spenders (the signal)")
    if contam.size:
        ax.hist(contam, bins=bins, color=CONTAM, alpha=0.85,
                label=f"impossible contamination (n={contam.size})")
    ax.set_xscale("log")
    q95 = np.quantile(vals, 0.95)
    q99 = np.quantile(vals, 0.99)
    ax.axvline(q95, color=WARN, linewidth=2.2, linestyle="--",
               label=f"winsor 95th = {q95:.0f} (aggressive: clips genuine spenders, disqualified)")
    ax.axvline(q99, color=WINNER, linewidth=2.2, linestyle="-",
               label=f"winsor 99th = {q99:.0f} (light: winner, removes only the extreme tail)")
    ax.set_xlabel("Revenue per user (USD/day, log scale)")
    ax.set_ylabel("Number of users")
    ax.set_title("Revenue: the treatment effect lives in the genuine spender tail.\n"
                 "Clip too hard (95th) and you delete the signal; clip lightly (99th) and you keep it while removing contamination",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper right")
    fig.tight_layout()
    path = os.path.join(io_utils.IMAGES_DIR, "revenue_signal_vs_contamination.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def early_vs_long_scatter(config, per_exp, winners):
    metric_names = list(config["metrics"].keys())
    fig, axes = plt.subplots(1, len(metric_names), figsize=(15, 4.8))
    for ax, metric_name in zip(axes, metric_names):
        winner = winners[metric_name]["method"]
        gw = per_exp[(per_exp["metric"] == metric_name) & (per_exp["method"] == winner)]
        gn = per_exp[(per_exp["metric"] == metric_name) & (per_exp["method"] == "none")]
        ax.scatter(gn["early_rel"] * 100, gn["long_rel"] * 100, s=14, color=WARN,
                   alpha=0.55, label="no handling")
        ax.scatter(gw["early_rel"] * 100, gw["long_rel"] * 100, s=14, color=WINNER,
                   alpha=0.7, label=f"winner: {winner}")
        lims = np.array([
            min(gn["early_rel"].min(), gw["early_rel"].min(), gn["long_rel"].min(), gw["long_rel"].min()),
            max(gn["early_rel"].max(), gw["early_rel"].max(), gn["long_rel"].max(), gw["long_rel"].max()),
        ]) * 100
        ax.plot(lims, lims, color=MUTED, linewidth=1, linestyle=":")
        ax.set_title(metric_name.replace("_", " "), fontsize=10)
        ax.set_xlabel("early reading (% change)")
        ax.set_ylabel("long-term reading (% change)")
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Predictiveness: does the early reading track the long-term reading (diagonal is perfect)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(io_utils.IMAGES_DIR, "early_vs_long_predictiveness.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def decision_quality(config, dec, winners):
    order = _method_order(config)
    metric_names = list(config["metrics"].keys())
    fig, axes = plt.subplots(1, len(metric_names), figsize=(20, 5.2), sharey=True)
    for ax, metric_name in zip(axes, metric_names):
        g = dec[dec["metric"] == metric_name].set_index("method").reindex(order)
        winner = winners[metric_name]["method"]
        colors = _bar_colors(order, winner)
        ax.bar(order, g["decision_error_rate"].to_numpy() * 100, color=colors)
        ax.set_title(metric_name.replace("_", " "), fontsize=10)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90, ha="center", fontsize=6.5)
        ax.set_ylabel("decision error rate (%)")
    fig.suptitle("Wrong ship / no-ship decisions by method (lower is better)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(io_utils.IMAGES_DIR, "decision_quality.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def decision_error_reduction(config, dec, winners):
    """Stakeholder-facing business-outcome view. For each metric, a grouped bar
    compares the ship / no-ship decision-error rate under no handling against the
    recommended per-metric method. Built from the committed decision table."""
    metric_names = list(config["metrics"].keys())
    short = {
        "revenue_per_user": "Revenue\nper user",
        "engagement_events_per_user": "Engagement\nevents per user",
        "conversion_rate_per_user": "Conversion\nrate per user",
    }
    none_rates, win_rates, win_methods = [], [], []
    for metric_name in metric_names:
        g = dec[dec["metric"] == metric_name].set_index("method")
        winner = winners[metric_name]["method"]
        none_rates.append(float(g.loc["none", "decision_error_rate"]) * 100)
        win_rates.append(float(g.loc[winner, "decision_error_rate"]) * 100)
        win_methods.append(winner)

    x = np.arange(len(metric_names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    b1 = ax.bar(x - width / 2, none_rates, width, color=WARN, label="no handling")
    b2 = ax.bar(x + width / 2, win_rates, width, color=WINNER,
                label="recommended per-metric method")

    for rects in (b1, b2):
        for r in rects:
            ax.annotate(f"{r.get_height():.1f}%", (r.get_x() + r.get_width() / 2, r.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        va="bottom", fontsize=9, color=INK)

    ymax = max(none_rates) if none_rates else 1.0
    for xi, none_v, win_v in zip(x, none_rates, win_rates):
        red = none_v - win_v
        label = "no change" if abs(red) < 0.05 else f"−{red:.1f} pp"
        ax.annotate(label, (xi, max(none_v, win_v)), textcoords="offset points",
                    xytext=(0, 18), ha="center", va="bottom", fontsize=9.5,
                    fontweight="bold", color=ACCENT)

    ax.set_xticks(x)
    ax.set_xticklabels([short[m] for m in metric_names], fontsize=9.5)
    ax.set_ylabel("ship / no-ship decision-error rate (%)")
    ax.set_ylim(0, ymax * 1.28)
    mean_red = float(np.mean([n - w for n, w in zip(none_rates, win_rates)]))
    ax.set_title("Business outcome: wrong ship / no-ship decisions drop with the "
                 "recommended per-metric method\n(lower is better, "
                 f"−{mean_red:.1f} pp on average vs no handling)",
                 fontsize=11.5, fontweight="bold")
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    fig.tight_layout()
    path = os.path.join(io_utils.IMAGES_DIR, "decision_error_reduction.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def calibration_gate_illustration(config, per_exp, path):
    """Drill-down on the calibration gate. For four example experiments, show the
    early 95 percent CI (shaded band) against the eventual long-term estimate
    (dashed line and marker with its own CI): covered, missed high, missed low, and
    a borderline case. This is exactly what the gate counts, per method, over all
    experiments. Uses no handling on the revenue metric, whose wide-tailed early
    reads produce all four cases."""
    metric = "revenue_per_user"
    g = per_exp[per_exp["metric"] == metric].reset_index(drop=True)
    lo = g["early_ci_low"].to_numpy(); hi = g["early_ci_high"].to_numpy()
    early = g["early_rel"].to_numpy(); long = g["long_rel"].to_numpy()
    llo = g["long_ci_low"].to_numpy(); lhi = g["long_ci_high"].to_numpy()
    meth = g["method"].to_numpy()

    half = (hi - lo) / 2.0
    center = (hi + lo) / 2.0
    zpos = (long - center) / np.where(half > 0, half, np.nan)  # 0=center, +/-1=boundary
    within = (lo <= long) & (long <= hi)
    idx = np.arange(len(g))

    def pick(mask, key):
        c = idx[mask]
        return int(c[np.argmin(key[c])]) if len(c) else None

    cases = [
        ("Early CI covers the eventual estimate", pick(within, np.abs(zpos)), True),
        ("Early CI misses high\n(eventual above the interval)", pick(long > hi, -zpos), False),
        ("Early CI misses low\n(eventual below the interval)", pick(long < lo, zpos), False),
        ("Borderline: eventual sits at the CI edge", pick(within, np.abs(1.0 - np.abs(zpos))), True),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    for ax, (title, i, ok) in zip(axes, cases):
        if i is None:
            ax.axis("off"); ax.set_title(title, fontsize=9); continue
        col = WINNER if ok else CONTAM
        title = f"{title}\n(method: {meth[i]})"
        el, eh, e = lo[i] * 100, hi[i] * 100, early[i] * 100
        L, Ll, Lh = long[i] * 100, llo[i] * 100, lhi[i] * 100
        ax.axhspan(el, eh, color=ACCENT, alpha=0.13, label="early 95% CI band")
        ax.errorbar([0], [e], yerr=[[e - el], [eh - e]], fmt="o", color=ACCENT,
                    capsize=4, label="early reading")
        ax.errorbar([1], [L], yerr=[[L - Ll], [Lh - L]], fmt="s", color=col,
                    capsize=4, label="eventual estimate")
        ax.axhline(L, color=col, lw=1, ls="--")
        ax.set_xlim(-0.5, 1.5); ax.set_xticks([0, 1]); ax.set_xticklabels(["early", "long"])
        ax.set_ylabel("relative effect (%)")
        ax.set_title(title, fontsize=9.5)
    axes[0].legend(fontsize=7.5, loc="best", frameon=False)
    fig.suptitle("What the calibration gate measures: does the early 95% CI cover the eventual estimate\n"
                 "(revenue_per_user, example experiments; the gate is the share of experiments in the covered case)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all(config, summary, winners, per_exp, dec, bank):
    io_utils.ensure_dirs()
    paths = []
    paths += per_metric_comparison(config, summary, winners)
    paths.append(composite_heatmap(config, summary, winners))
    paths.append(revenue_signal_illustration(config, bank))
    paths.append(early_vs_long_scatter(config, per_exp, winners))
    paths.append(calibration_gate_illustration(config, per_exp,
                 os.path.join(io_utils.IMAGES_DIR, "calibration_gate_cases.png")))
    paths.append(decision_quality(config, dec, winners))
    paths.append(decision_error_reduction(config, dec, winners))
    return paths
