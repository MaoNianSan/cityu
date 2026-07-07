"""Publication-oriented plotting from aggregate result tables only.

Figure contract
---------------
1. Main figures: one 95% confidence-level figure per scenario.  The left panel
   reports empirical coverage for all methods; the right panel reports relative
   CI width only for methods that retain a valid-inference interpretation
   (Classic, PPI, and PPI++).  Naive ML remains visible in the coverage panel
   because its failure is part of the diagnostic, but it is excluded from the
   efficiency comparison because near-zero invalid intervals are not evidence
   of inferential efficiency.
2. Calibration figures: one appendix-oriented figure per scenario.  It reports
   empirical coverage minus nominal coverage at all configured confidence
   levels.  Each profile occupies one column, which keeps P2/P3 (the
   matched-MSE structural contrast) explicit.

The plotting code never reads raw simulated data and never recomputes an
estimate or confidence interval.  It works exclusively from aggregate metric
tables produced by ``main.py``.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from formulation import display_name, get_scenario


METHOD_LABELS = {
    "classic": "Classic",
    "naive_ml": "Naive ML",
    "ppi": "PPI",
    "ppi_plus_plus": "PPI++",
}
PROFILE_LABELS = {
    "P1": "P1\nsmall random",
    "P2": "P2\nconstant shift",
    "P3": "P3\ncovariate-linked",
    "P4": "P4\nlarge random",
}
PROFILE_SHORT_LABELS = {profile: profile for profile in config.ACTIVE_PROFILES}
COVERAGE_METHODS = ("classic", "naive_ml", "ppi", "ppi_plus_plus")
EFFICIENCY_METHODS = ("ppi", "ppi_plus_plus")
PROFILE_OFFSETS = {"naive_ml": -0.18, "ppi": -0.06, "ppi_plus_plus": 0.10}
CALIBRATION_MARKERS = {
    "classic": "s",
    "naive_ml": "x",
    "ppi": "o",
    "ppi_plus_plus": "^",
}


def _single_row(frame: pd.DataFrame, **conditions: object) -> pd.Series:
    selected = frame
    for key, value in conditions.items():
        selected = selected[selected[key] == value]
    if len(selected) != 1:
        raise KeyError(f"Expected one row for {conditions}, found {len(selected)}.")
    return selected.iloc[0]


def _build_seed_level_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add width relative to the matched Classic result within each seed."""
    enriched = frame.copy()
    classic = (
        enriched[enriched["method"] == "classic"]
        [["seed", "scenario", "target", "confidence_level", "average_ci_width"]]
        .rename(columns={"average_ci_width": "classic_average_ci_width"})
    )
    enriched = enriched.merge(
        classic,
        on=["seed", "scenario", "target", "confidence_level"],
        how="left",
        validate="many_to_one",
    )
    if enriched["classic_average_ci_width"].isna().any():
        raise ValueError("Every method/profile cell must have a matched Classic CI width.")
    if (enriched["classic_average_ci_width"] <= 0.0).any():
        raise ValueError("Classic average CI widths must be strictly positive.")

    enriched["relative_ci_width"] = (
        enriched["average_ci_width"] / enriched["classic_average_ci_width"]
    )
    enriched.loc[enriched["method"] == "classic", "relative_ci_width"] = 1.0
    enriched["coverage_error"] = enriched["empirical_coverage"] - enriched["confidence_level"]
    return enriched


def _summarise_for_plot(frame: pd.DataFrame) -> pd.DataFrame:
    """Return median/IQR seed summaries; fast mode becomes a one-seed summary."""
    enriched = _build_seed_level_metrics(frame)
    key_columns = ["scenario", "target", "profile", "method", "confidence_level"]
    summary = (
        enriched.groupby(key_columns, as_index=False)
        .agg(
            center_empirical_coverage=("empirical_coverage", "median"),
            q25_empirical_coverage=("empirical_coverage", lambda x: x.quantile(0.25)),
            q75_empirical_coverage=("empirical_coverage", lambda x: x.quantile(0.75)),
            center_relative_ci_width=("relative_ci_width", "median"),
            q25_relative_ci_width=("relative_ci_width", lambda x: x.quantile(0.25)),
            q75_relative_ci_width=("relative_ci_width", lambda x: x.quantile(0.75)),
            center_coverage_error=("coverage_error", "median"),
            q25_coverage_error=("coverage_error", lambda x: x.quantile(0.25)),
            q75_coverage_error=("coverage_error", lambda x: x.quantile(0.75)),
            n_seeds=("seed", "nunique"),
            n_replicates=("n_replicates", "first"),
        )
        .sort_values(key_columns)
        .reset_index(drop=True)
    )
    return summary


def _coverage_reference_band(confidence_level: float, n_replicates: int) -> tuple[float, float]:
    """A 95% binomial Monte-Carlo reference range under exact calibration."""
    standard_error = np.sqrt(confidence_level * (1.0 - confidence_level) / n_replicates)
    half_width = 1.96 * standard_error
    return max(0.0, confidence_level - half_width), min(1.0, confidence_level + half_width)


def _summary_note(n_seeds: int, n_replicates: int) -> str:
    if n_seeds == 1:
        return f"Fast diagnostic: seed 0 only; R={n_replicates} replicates."
    return (
        f"Full result: markers are medians across {n_seeds} outer seeds; "
        f"vertical bars are cross-seed IQRs; R={n_replicates} replicates per seed."
    )


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    saved: list[Path] = []
    for suffix in config.SAVE_FIGURE_FORMATS:
        path = output_dir / f"{stem}.{suffix}"
        kwargs = {"dpi": config.FIGURE_DPI} if suffix == "png" else {}
        fig.savefig(path, bbox_inches="tight", **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved


def _draw_iqr(ax: plt.Axes, x_value: float, q25: float, q75: float, color: str) -> None:
    if np.isfinite(q25) and np.isfinite(q75) and q25 < q75:
        ax.vlines(x_value, q25, q75, color=color, linewidth=2.6, alpha=0.35, zorder=2)


def _draw_main_coverage_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    scenario: str,
    target: str,
    confidence_level: float,
) -> None:
    profiles = list(config.ACTIVE_PROFILES)
    x_base = np.arange(len(profiles), dtype=float)
    n_replicates = int(
        _single_row(
            summary,
            scenario=scenario,
            target=target,
            profile="baseline",
            method="classic",
            confidence_level=confidence_level,
        )["n_replicates"]
    )
    band_low, band_high = _coverage_reference_band(confidence_level, n_replicates)
    ax.axhspan(band_low, band_high, color="0.85", alpha=0.70, zorder=0)
    ax.axhline(confidence_level, color="0.35", linestyle=":", linewidth=1.2, label="Nominal")

    classic = _single_row(
        summary,
        scenario=scenario,
        target=target,
        profile="baseline",
        method="classic",
        confidence_level=confidence_level,
    )
    classic_center = float(classic["center_empirical_coverage"])
    classic_q25 = float(classic["q25_empirical_coverage"])
    classic_q75 = float(classic["q75_empirical_coverage"])
    classic_line = ax.axhline(
        classic_center,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Classic",
    )
    if classic_q25 < classic_q75:
        ax.axhspan(classic_q25, classic_q75, color=classic_line.get_color(), alpha=0.08, zorder=0)

    for method in ("naive_ml", "ppi", "ppi_plus_plus"):
        x_values: list[float] = []
        y_values: list[float] = []
        q25_values: list[float] = []
        q75_values: list[float] = []
        for profile_index, profile in enumerate(profiles):
            row = _single_row(
                summary,
                scenario=scenario,
                target=target,
                profile=profile,
                method=method,
                confidence_level=confidence_level,
            )
            x_values.append(x_base[profile_index] + PROFILE_OFFSETS[method])
            y_values.append(float(row["center_empirical_coverage"]))
            q25_values.append(float(row["q25_empirical_coverage"]))
            q75_values.append(float(row["q75_empirical_coverage"]))

        marker = "x" if method == "naive_ml" else "o"
        line, = ax.plot(
            x_values,
            y_values,
            marker=marker,
            linestyle="None",
            markersize=6.5,
            alpha=0.76 if method == "naive_ml" else 0.96,
            label=METHOD_LABELS[method],
            zorder=3,
        )
        for x_value, q25, q75 in zip(x_values, q25_values, q75_values):
            _draw_iqr(ax, x_value, q25, q75, line.get_color())

    ax.set_xticks(x_base)
    ax.set_xticklabels([PROFILE_LABELS[profile] for profile in profiles])
    ax.set_ylim(*config.PLOT_COVERAGE_YLIM)
    ax.set_ylabel("Empirical coverage")
    ax.grid(axis="y", alpha=0.25)


def _draw_main_efficiency_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    scenario: str,
    target: str,
    confidence_level: float,
) -> None:
    profiles = list(config.ACTIVE_PROFILES)
    x_base = np.arange(len(profiles), dtype=float)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="_nolegend_")
    ax.text(0.98, 1.0, "Classic = 1", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8.5)

    upper_candidates = [1.0]
    for method in EFFICIENCY_METHODS:
        x_values: list[float] = []
        y_values: list[float] = []
        q25_values: list[float] = []
        q75_values: list[float] = []
        for profile_index, profile in enumerate(profiles):
            row = _single_row(
                summary,
                scenario=scenario,
                target=target,
                profile=profile,
                method=method,
                confidence_level=confidence_level,
            )
            x_values.append(x_base[profile_index] + PROFILE_OFFSETS[method])
            y_values.append(float(row["center_relative_ci_width"]))
            q25_values.append(float(row["q25_relative_ci_width"]))
            q75_values.append(float(row["q75_relative_ci_width"]))
            upper_candidates.append(float(row["q75_relative_ci_width"]))

        line, = ax.plot(
            x_values,
            y_values,
            marker="o",
            linestyle="None",
            markersize=6.5,
            label=METHOD_LABELS[method],
            zorder=3,
        )
        for x_value, q25, q75 in zip(x_values, q25_values, q75_values):
            _draw_iqr(ax, x_value, q25, q75, line.get_color())

    ax.set_xticks(x_base)
    ax.set_xticklabels([PROFILE_LABELS[profile] for profile in profiles])
    ax.set_ylim(0.0, max(1.15, 1.08 * max(upper_candidates)))
    ax.set_ylabel("Relative CI width\n(vs Classic)")
    ax.grid(axis="y", alpha=0.25)


def _plot_main_figure(
    summary: pd.DataFrame,
    output_dir: Path,
    scenario_name: str,
    confidence_level: float,
    summary_note: str,
) -> list[Path]:
    scenario = get_scenario(scenario_name)
    row_count = len(scenario.target_names)
    fig, axes = plt.subplots(
        row_count,
        2,
        figsize=(12.0, max(4.0, 3.75 * row_count)),
        squeeze=False,
    )
    confidence_text = f"{confidence_level * 100:.1f}%"
    fig.suptitle(
        f"{display_name(scenario_name)}: inference validity and efficiency ({confidence_text} confidence)",
        fontsize=14,
        y=0.985,
    )

    for row_index, target in enumerate(scenario.target_names):
        coverage_ax = axes[row_index, 0]
        efficiency_ax = axes[row_index, 1]
        _draw_main_coverage_panel(coverage_ax, summary, scenario_name, target, confidence_level)
        _draw_main_efficiency_panel(efficiency_ax, summary, scenario_name, target, confidence_level)
        target_title = target.replace("_", r"\_")
        coverage_ax.set_title(f"{target_title}: coverage")
        efficiency_ax.set_title(f"{target_title}: valid-method efficiency")

    coverage_handles, coverage_labels = axes[0, 0].get_legend_handles_labels()
    efficiency_handles, efficiency_labels = axes[0, 1].get_legend_handles_labels()
    legend_map = dict(zip(coverage_labels + efficiency_labels, coverage_handles + efficiency_handles))
    fig.legend(
        legend_map.values(),
        legend_map.keys(),
        ncol=min(5, len(legend_map)),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        frameon=False,
    )
    fig.text(
        0.5,
        0.045,
        "Grey band: 95% binomial Monte-Carlo reference range under exact calibration. "
        "Naive ML is omitted from relative-width panels because invalid coverage is not efficiency.\n"
        + summary_note,
        ha="center",
        va="bottom",
        fontsize=8.8,
    )
    fig.tight_layout(rect=(0.0, 0.095, 1.0, 0.90))
    return _save_figure(
        fig,
        output_dir,
        f"{scenario_name}_main_cl_{int(round(confidence_level * 1000)):03d}",
    )


def _draw_calibration_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    scenario: str,
    target: str,
    profile: str,
    y_limits: tuple[float, float],
) -> None:
    """Draw coverage error by nominal level, censoring only off-scale failures.

    The shared y-axis is deliberately focused on plausible calibration error.
    A near-zero-width Naive-ML interval can produce coverage error near -1;
    showing that value on the same linear scale would conceal whether PPI and
    PPI++ are calibrated.  Such points are therefore marked with a downward
    triangle at the display boundary.  The exact, uncensored values are kept
    in ``table/plot_summary.csv`` and the primary coverage figures.
    """
    levels = np.asarray(config.CONFIDENCE_LEVELS, dtype=float)
    n_replicates = int(
        _single_row(
            summary,
            scenario=scenario,
            target=target,
            profile="baseline",
            method="classic",
            confidence_level=float(levels[0]),
        )["n_replicates"]
    )
    lower_limit, upper_limit = y_limits
    ax.axhline(0.0, color="0.35", linestyle=":", linewidth=1.1, label="Exact calibration")

    for method in COVERAGE_METHODS:
        centers: list[float] = []
        q25s: list[float] = []
        q75s: list[float] = []
        x_values: list[float] = []
        for level in levels:
            source_profile = "baseline" if method == "classic" else profile
            row = _single_row(
                summary,
                scenario=scenario,
                target=target,
                profile=source_profile,
                method=method,
                confidence_level=float(level),
            )
            x_values.append(float(level))
            centers.append(float(row["center_coverage_error"]))
            q25s.append(float(row["q25_coverage_error"]))
            q75s.append(float(row["q75_coverage_error"]))

        x_array = np.asarray(x_values, dtype=float)
        center_array = np.asarray(centers, dtype=float)
        q25_array = np.asarray(q25s, dtype=float)
        q75_array = np.asarray(q75s, dtype=float)
        within = (center_array >= lower_limit) & (center_array <= upper_limit)
        lower_censored = center_array < lower_limit
        upper_censored = center_array > upper_limit

        style = "--" if method == "classic" else "-"
        alpha = 0.75 if method == "naive_ml" else 0.96
        # Establish one stable legend/color handle per method, then draw only
        # in-range points as a conventional line.
        handle, = ax.plot(
            [],
            [],
            marker=CALIBRATION_MARKERS[method],
            linestyle=style,
            linewidth=1.2,
            markersize=5.6,
            alpha=alpha,
            label=METHOD_LABELS[method],
        )
        color = handle.get_color()
        if within.any():
            ax.plot(
                x_array[within],
                center_array[within],
                marker=CALIBRATION_MARKERS[method],
                linestyle=style,
                linewidth=1.2,
                markersize=5.6,
                alpha=alpha,
                color=color,
                zorder=3,
            )
            for x_value, q25, q75 in zip(
                x_array[within], q25_array[within], q75_array[within]
            ):
                _draw_iqr(
                    ax,
                    float(x_value),
                    max(lower_limit, float(q25)),
                    min(upper_limit, float(q75)),
                    color,
                )
        if lower_censored.any():
            ax.plot(
                x_array[lower_censored],
                np.full(lower_censored.sum(), lower_limit + 0.006),
                marker="v",
                linestyle="None",
                markersize=6.2,
                alpha=alpha,
                color=color,
                zorder=4,
            )
        if upper_censored.any():
            ax.plot(
                x_array[upper_censored],
                np.full(upper_censored.sum(), upper_limit - 0.006),
                marker="^",
                linestyle="None",
                markersize=6.2,
                alpha=alpha,
                color=color,
                zorder=4,
            )

    for level in levels:
        lower, upper = _coverage_reference_band(float(level), n_replicates)
        ax.vlines(
            float(level),
            max(lower_limit, lower - float(level)),
            min(upper_limit, upper - float(level)),
            color="0.75",
            linewidth=4.0,
            alpha=0.75,
            zorder=1,
        )

    ax.set_xlim(float(levels.min()) - 0.004, float(levels.max()) + 0.004)
    ax.set_xticks(levels)
    ax.set_xticklabels([f"{level * 100:.1f}%" for level in levels])
    ax.set_ylim(*y_limits)
    ax.grid(axis="y", alpha=0.25)


def _plot_calibration_figure(
    summary: pd.DataFrame,
    output_dir: Path,
    scenario_name: str,
    summary_note: str,
) -> list[Path]:
    scenario = get_scenario(scenario_name)
    profiles = list(config.ACTIVE_PROFILES)
    target_count = len(scenario.target_names)
    y_limits = tuple(float(value) for value in config.CALIBRATION_COVERAGE_ERROR_YLIM)
    fig, axes = plt.subplots(
        target_count,
        len(profiles),
        figsize=(3.25 * len(profiles), max(3.5, 2.85 * target_count)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    fig.suptitle(
        f"{display_name(scenario_name)}: coverage calibration across nominal confidence levels",
        fontsize=14,
        y=0.99,
    )

    for row_index, target in enumerate(scenario.target_names):
        for column_index, profile in enumerate(profiles):
            ax = axes[row_index, column_index]
            _draw_calibration_panel(ax, summary, scenario_name, target, profile, y_limits)
            if row_index == 0:
                ax.set_title(PROFILE_SHORT_LABELS[profile])
            if column_index == 0:
                target_label = target.replace("_", r"\_")
                ax.set_ylabel(f"{target_label}\nCoverage − nominal")
            if row_index == target_count - 1:
                ax.set_xlabel("Nominal confidence")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_map = dict(zip(labels, handles))
    fig.legend(
        legend_map.values(),
        legend_map.keys(),
        ncol=min(5, len(legend_map)),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        frameon=False,
    )
    fig.text(
        0.5,
        0.028,
        "Zero denotes exact calibration. Grey vertical bars are 95% binomial Monte-Carlo reference ranges. "
        "Downward/upward triangles indicate coverage error outside the displayed range.\n"
        "P2 and P3 have matched pointwise proxy MSE but different error structure. "
        + summary_note,
        ha="center",
        va="bottom",
        fontsize=8.6,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.89))
    return _save_figure(fig, output_dir, f"{scenario_name}_calibration")


def _remove_stale_generated_figures(output_dir: Path) -> None:
    """Remove only rendered files; keep .gitkeep or user documentation intact."""
    for suffix in config.SAVE_FIGURE_FORMATS:
        for path in output_dir.glob(f"*.{suffix}"):
            path.unlink()


def _write_plot_summary(summary: pd.DataFrame, output_dir: Path) -> Path:
    table_dir = output_dir.parent / "table"
    table_dir.mkdir(parents=True, exist_ok=True)
    destination = table_dir / "plot_summary.csv"
    summary.to_csv(destination, index=False)
    return destination


def plot_all_results(
    seed0_metrics: pd.DataFrame,
    output_dir: Path,
    robustness_summary: pd.DataFrame | None = None,
    all_metrics: pd.DataFrame | None = None,
) -> list[Path]:
    """Write the main 95% figures and calibration appendix figures.

    ``robustness_summary`` remains in the signature for backward compatibility;
    all full-mode plotting derives directly from ``all_metrics`` so medians and
    IQRs are computed from the same outer-seed population.
    """
    del robustness_summary
    output_dir.mkdir(parents=True, exist_ok=True)
    _remove_stale_generated_figures(output_dir)

    source_metrics = all_metrics if all_metrics is not None and not all_metrics.empty else seed0_metrics
    summary = _summarise_for_plot(source_metrics)
    _write_plot_summary(summary, output_dir)

    n_replicates = int(source_metrics["n_replicates"].iloc[0])
    n_seeds = int(source_metrics["seed"].nunique())
    summary_note = _summary_note(n_seeds=n_seeds, n_replicates=n_replicates)

    main_level = float(config.MAIN_FIGURE_CONFIDENCE_LEVEL)
    if main_level not in {float(level) for level in config.CONFIDENCE_LEVELS}:
        raise ValueError("MAIN_FIGURE_CONFIDENCE_LEVEL must be present in CONFIDENCE_LEVELS.")

    saved: list[Path] = []
    for scenario_name in config.ACTIVE_SCENARIOS:
        saved.extend(_plot_main_figure(summary, output_dir, scenario_name, main_level, summary_note))
        saved.extend(_plot_calibration_figure(summary, output_dir, scenario_name, summary_note))
    return saved
