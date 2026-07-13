from __future__ import annotations
import re
from pathlib import Path
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

plt.ioff()
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from config import (
    EXPERIMENT_LABELS,
    LEARNER_IDS,
    LEARNER_LABELS,
    METHOD_LABELS,
    PARAMETER_LABELS,
)

METHODS = ["ppi", "ppi_plus_plus_v1", "ppi_plus_plus_v2", "cross_ppi"]
CAL_METHODS = ["classic", *METHODS]
MARKERS = {
    "classic": "X",
    "ppi": "o",
    "ppi_plus_plus_v1": "s",
    "ppi_plus_plus_v2": "D",
    "cross_ppi": "^",
}
COLORS = {
    "classic": "0.1",
    "ppi": "#1f77b4",
    "ppi_plus_plus_v1": "#ff7f0e",
    "ppi_plus_plus_v2": "#2ca02c",
    "cross_ppi": "#d62728",
}
OFFSETS = dict(zip(METHODS, [-0.18, -0.06, 0.06, 0.18]))
CAL_OFFSETS = dict(zip(CAL_METHODS, [-0.24, -0.12, 0, 0.12, 0.24]))


def safe_parameter(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_")


def _save(fig, outdir, stem):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plabel(p):
    return PARAMETER_LABELS.get(p, str(p).replace("_", " ").replace("-", " ").title())


def _legend(methods=METHODS):
    return [
        Line2D(
            [0],
            [0],
            marker=MARKERS[m],
            color=COLORS[m],
            linestyle="none",
            label=METHOD_LABELS[m],
        )
        for m in methods
    ]


def plot_learner_quality(quality_df, outdir, experiment, n_replicates=None):
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    rng = np.random.default_rng(20260709)
    n_replicates = int(n_replicates or quality_df.replicate.nunique())
    alpha = 0.55 if n_replicates <= 30 else 0.25
    scale = 1e9 if experiment == "b_lr" else 1.0
    groups = []
    means = []
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, lid in enumerate(LEARNER_IDS):
        vals = (
            quality_df.loc[quality_df.learner == lid, "quality_value"]
            .dropna()
            .to_numpy(float)
            / scale
        )
        groups.append(vals if len(vals) else np.array([np.nan]))
        means.append(np.mean(vals) if len(vals) else np.nan)
        if len(vals):
            ax.scatter(
                i + rng.uniform(-0.09, 0.09, len(vals)),
                vals,
                s=20,
                alpha=alpha,
                color=colors[i],
                edgecolors="none",
                zorder=3,
            )
    boxes = ax.boxplot(
        groups, positions=np.arange(4), widths=0.52, showfliers=False, patch_artist=True
    )
    for p, c in zip(boxes["boxes"], colors):
        p.set(facecolor=to_rgba(c, 0.16), edgecolor=c)
    for k in ("whiskers", "caps", "medians"):
        for a in boxes[k]:
            a.set_color(".35")
    finite = np.concatenate([g[np.isfinite(g)] for g in groups if np.isfinite(g).any()])
    lo, hi = float(finite.min()), float(finite.max())
    span = max(hi - lo, abs(hi) * 0.05, 1e-9)
    ax.set_ylim(lo - 0.08 * span, hi + 0.25 * span)
    for i, m in enumerate(means):
        if np.isfinite(m):
            ax.scatter(i, m, marker="D", s=45, color=".08", zorder=5)
            ax.text(
                i,
                hi + 0.08 * span,
                f"{m:.3f}" if experiment == "b_lr" else f"{m:.4f}".rstrip("0"),
                ha="center",
                fontsize=9,
            )
    ax.set_xticks(range(4), [LEARNER_LABELS[x] for x in LEARNER_IDS])
    ax.set_ylabel(
        "Prediction MSE (×10⁹; lower is better)"
        if experiment == "b_lr"
        else "Brier score (lower is better)"
    )
    ax.set_title("Learner prediction quality", fontsize=14, pad=28)
    ax.text(
        0.5,
        1.025,
        EXPERIMENT_LABELS[experiment],
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color=".35",
    )
    ax.grid(axis="y", alpha=0.22)
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.14, right=0.97)
    _save(fig, outdir, "learner_quality")


def plot_inference_performance_95(
    summary_95, quality_df, outdir, experiment, n_replicates
):
    params = list(dict.fromkeys(summary_95.parameter))
    B = int(n_replicates)
    h = 1.96 * np.sqrt(0.95 * 0.05 / B)
    xs = np.arange(4)
    vals = (
        summary_95[summary_95.method.isin(["classic", *METHODS])]
        .coverage_error.dropna()
        .to_numpy(float)
    )
    raw = max([h, *np.abs(vals).tolist()])
    vlim = max(h * 1.15, raw + max(0.025, 0.15 * raw))
    eff = []
    for c in (
        "relative_width_rep_median",
        "relative_width_rep_q25",
        "relative_width_rep_q75",
    ):
        eff.extend(summary_95.loc[summary_95.method.isin(METHODS), c].dropna().tolist())
    emin, emax = min(eff), max(eff)
    span = emax - emin
    margin = max(0.05, 0.1 * span)
    elo = max(0, emin - margin)
    ehi = emax + margin
    if ehi - elo < 0.2:
        mid = (elo + ehi) / 2
        elo = max(0, mid - 0.1)
        ehi = elo + 0.2
    fig, axes = plt.subplots(
        len(params),
        2,
        figsize=(13, 3.25 * len(params) + 2.3),
        squeeze=False,
        sharex="col",
    )
    qmeans = quality_df.groupby("learner").quality_value.mean()
    qscale = 1e9 if experiment == "b_lr" else 1
    qlabels = [
        f"{LEARNER_LABELS[l]}\n{qmeans.get(l,np.nan)/qscale:.3f}" for l in LEARNER_IDS
    ]
    for r, p in enumerate(params):
        sub = summary_95[summary_95.parameter == p]
        left, right = axes[r]
        left.axhspan(-h, h, color=".7", alpha=0.10)
        left.axhline(-h, color=".6", ls="--", lw=0.7)
        left.axhline(h, color=".6", ls="--", lw=0.7)
        left.axhline(0, color=".35", ls="--", lw=1)
        classic = sub[sub.method == "classic"].coverage_error.dropna()
        ce = float(classic.iloc[0]) if len(classic) else np.nan
        if np.isfinite(ce):
            left.axhline(ce, color=".1", ls="-.", lw=1)
            left.text(
                3.38,
                ce,
                f"Classic: {ce:+.2f}".replace("-", "−"),
                va="bottom",
                ha="right",
                fontsize=8,
            )
        right.axhline(1, color=".4", ls="--", lw=1)
        right.text(
            3.38, 1, "Classic width", va="bottom", ha="right", fontsize=8, color=".35"
        )
        for m in METHODS:
            for i, l in enumerate(LEARNER_IDS):
                row = sub[(sub.method == m) & (sub.learner == l)]
                if row.empty:
                    continue
                y = float(row.coverage_error.iloc[0])
                left.scatter(
                    i + OFFSETS[m],
                    y,
                    marker=MARKERS[m],
                    color=COLORS[m],
                    s=48,
                    zorder=4,
                )
                med = float(row.relative_width_rep_median.iloc[0])
                q1 = float(row.relative_width_rep_q25.iloc[0])
                q3 = float(row.relative_width_rep_q75.iloc[0])
                right.errorbar(
                    i + OFFSETS[m],
                    med,
                    yerr=[[med - q1], [q3 - med]],
                    fmt=MARKERS[m],
                    color=COLORS[m],
                    markerfacecolor=COLORS[m] if abs(y) <= h else "none",
                    markersize=6,
                    capsize=3,
                    lw=1.1,
                    zorder=4,
                )
        left.set_ylim(-vlim, vlim)
        right.set_ylim(elo, ehi)
        left.set_ylabel(_plabel(p))
        left.grid(axis="y", alpha=0.2)
        right.grid(axis="y", alpha=0.2)
        if r < len(params) - 1:
            left.tick_params(labelbottom=False)
            right.tick_params(labelbottom=False)
        else:
            left.set_xticks(xs, qlabels)
            right.set_xticks(xs, qlabels)
            lab = (
                "Learner / mean MSE in ×10⁹ units (↓)"
                if experiment == "b_lr"
                else "Learner / mean Brier score (↓)"
            )
            left.set_xlabel(lab)
            right.set_xlabel(lab)
    axes[0, 0].set_title("Validity: coverage error", pad=10)
    axes[0, 1].set_title("Efficiency: relative CI width", pad=10)
    fig.suptitle(
        "95% inference performance", y=0.985, fontsize=15, fontweight="semibold"
    )
    fig.text(
        0.5, 0.953, EXPERIMENT_LABELS[experiment], ha="center", fontsize=10, color=".35"
    )
    fig.legend(
        handles=_legend(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.012,
        "Hollow markers indicate 95% coverage outside the nominal Monte Carlo band.",
        ha="center",
        fontsize=9,
        color=".3",
    )
    fig.text(0.02, 0.012, "Shading: Nominal-coverage MC band", fontsize=8, color=".4")
    fig.subplots_adjust(
        top=0.84, bottom=0.13, left=0.13, right=0.98, hspace=0.30, wspace=0.25
    )
    _save(fig, outdir, "inference_performance_95")


def plot_coverage_calibration_by_parameter(
    summary_by_confidence, outdir, experiment, main_parameters, n_replicates
):
    data = summary_by_confidence[
        summary_by_confidence.parameter.isin(main_parameters)
        & summary_by_confidence.method.isin(CAL_METHODS)
    ]
    levels = sorted(data.confidence_level.unique())
    B = int(n_replicates)
    bands = {x: 1.96 * np.sqrt(x * (1 - x) / B) for x in levels}
    raw = max([*data.coverage_error.abs().dropna().tolist(), *bands.values()])
    ylim = max(0.08, raw + max(0.02, 0.12 * raw))
    x = np.arange(len(levels))
    for p in main_parameters:
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True)
        for c, l in enumerate(LEARNER_IDS):
            ax = axes[c]
            sub = data[
                (data.parameter == p)
                & ((data.learner == l) | (data.method == "classic"))
            ]
            for j, nom in enumerate(levels):
                ax.fill_between(
                    [j - 0.34, j + 0.34],
                    [-bands[nom]] * 2,
                    [bands[nom]] * 2,
                    color=".7",
                    alpha=0.12,
                )
            ax.axhline(0, color=".4", ls="--", lw=1)
            for m in CAL_METHODS:
                sm = sub[sub.method == m]
                sm = sm if m == "classic" else sm[sm.learner == l]
                for j, nom in enumerate(levels):
                    row = sm[np.isclose(sm.confidence_level, nom)]
                    if len(row):
                        ax.scatter(
                            j + CAL_OFFSETS[m],
                            float(row.coverage_error.iloc[0]),
                            marker=MARKERS[m],
                            color=COLORS[m],
                            s=42,
                            zorder=3,
                        )
            ax.set_title(LEARNER_LABELS[l])
            ax.set_xticks(x, [f"{100*z:g}%" for z in levels])
            ax.set_ylim(-ylim, ylim)
            ax.grid(axis="y", alpha=0.18)
            if c == 0:
                ax.set_ylabel("Coverage − nominal")
            ax.set_xlabel("Nominal level")
        fig.suptitle(f"Coverage calibration: {_plabel(p)}", y=0.98, fontsize=14)
        fig.text(
            0.5,
            0.92,
            EXPERIMENT_LABELS[experiment],
            ha="center",
            fontsize=9.5,
            color=".35",
        )
        fig.legend(
            handles=_legend(CAL_METHODS),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.86),
            ncol=5,
            frameon=False,
        )
        fig.subplots_adjust(top=0.72, bottom=0.18, left=0.07, right=0.99, wspace=0.12)
        _save(fig, outdir, f"diagnostic_coverage_calibration_{safe_parameter(p)}")
