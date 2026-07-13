"""Publication-oriented plotting from aggregate result tables only.

Figure contract
---------------
1. Main figures: one 95% confidence-level figure per scenario.  The left panel
   reports empirical coverage for all methods; the right panel reports relative
   CI width only for methods that retain a valid-inference interpretation
   (Classic, PPI, PPI++V1, and PPI++V2).  Naive ML remains visible in the coverage panel
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import config
from formulation import display_name, get_scenario


METHOD_LABELS = {
    "classic": "Classic",
    "naive_ml": "Naive ML",
    "ppi": "PPI",
    "ppi_plus_plus_v1": "PPI++V1",
    "ppi_plus_plus_v2": "PPI++V2",
}
METHOD_ORDER = (
    "classic",
    "naive_ml",
    "ppi",
    "ppi_plus_plus_v1",
    "ppi_plus_plus_v2",
)
METHOD_COLORS = {
    "classic": "black",
    "naive_ml": "0.45",
    "ppi": "#1f77b4",
    "ppi_plus_plus_v1": "#2ca02c",
    "ppi_plus_plus_v2": "#d62728",
}
PROFILE_LABELS = {
    "P1": "P1\nsmall random",
    "P2": "P2\nconstant shift",
    "P3": "P3\ncovariate-linked",
    "P4": "P4\nlarge random",
}
PROFILE_SHORT_LABELS = {profile: profile for profile in config.ACTIVE_PROFILES}
COVERAGE_METHODS = METHOD_ORDER
PROFILED_COVERAGE_METHODS = (
    "naive_ml",
    "ppi",
    "ppi_plus_plus_v1",
    "ppi_plus_plus_v2",
)
EFFICIENCY_METHODS = ("ppi", "ppi_plus_plus_v1", "ppi_plus_plus_v2")
CALIBRATION_MARKERS = {
    "classic": "s",
    "naive_ml": "x",
    "ppi": "o",
    "ppi_plus_plus_v1": "^",
    "ppi_plus_plus_v2": "D",
}
FOCUSED_HARD_YLIM = (0.75, 1.25)
FOCUSED_Y_PADDING = 0.10
FOCUSED_MIN_SPAN = 0.18
FOCUSED_MARKER_PAD_FRAC = 0.025


def _single_row(frame: pd.DataFrame, **conditions: object) -> pd.Series:
    selected = frame
    for key, value in conditions.items():
        selected = selected[selected[key] == value]
    if len(selected) != 1:
        raise KeyError(f"Expected one row for {conditions}, found {len(selected)}.")
    return selected.iloc[0]


def _maybe_single_row(frame: pd.DataFrame, **conditions: object) -> pd.Series | None:
    selected = frame
    for key, value in conditions.items():
        selected = selected[selected[key] == value]
    if len(selected) == 0:
        return None
    if len(selected) != 1:
        raise KeyError(f"Expected at most one row for {conditions}, found {len(selected)}.")
    return selected.iloc[0]


def _profile_offsets(methods: tuple[str, ...]) -> dict[str, float]:
    if not methods:
        return {}
    centered = np.arange(len(methods), dtype=float) - 0.5 * (len(methods) - 1)
    return {method: float(0.18 * offset) for method, offset in zip(methods, centered)}


def _method_legend_handles(methods: tuple[str, ...] | list[str]) -> list[Line2D]:
    handles: list[Line2D] = []

    for method in methods:
        if method == "classic":
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=METHOD_COLORS[method],
                    linestyle="--",
                    linewidth=1.5,
                    label=METHOD_LABELS[method],
                )
            )
        else:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=METHOD_COLORS[method],
                    marker=CALIBRATION_MARKERS.get(method, "o"),
                    linestyle="None",
                    markersize=6.5,
                    label=METHOD_LABELS[method],
                )
            )

    return handles


def _calibration_legend_handles() -> list[Line2D]:
    handles: list[Line2D] = [
        Line2D(
            [0],
            [0],
            color="0.35",
            linestyle=":",
            linewidth=1.1,
            label="Exact calibration",
        )
    ]

    for method in METHOD_ORDER:
        linestyle = "--" if method == "classic" else "-"
        handles.append(
            Line2D(
                [0],
                [0],
                color=METHOD_COLORS[method],
                marker=CALIBRATION_MARKERS.get(method, "o"),
                linestyle=linestyle,
                markersize=5.6,
                label=METHOD_LABELS[method],
            )
        )

    return handles


def _build_seed_level_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Add width relative to the matched Classic result within each seed."""
    enriched = frame.copy()
    if "n_failed_replicates" in enriched.columns:
        failures = pd.to_numeric(
            enriched["n_failed_replicates"], errors="coerce"
        ).fillna(0)
        if (failures > 0).any():
            failed_rows = enriched.loc[
                failures > 0,
                ["seed", "scenario", "target", "profile", "method", "n_failed_replicates"],
            ]
            raise ValueError(
                "Plotting refuses metrics with failed replications. Inspect the "
                f"diagnostic table first. Example failures: {failed_rows.head(5).to_dict('records')}"
            )
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
    """Return seed-level median, IQR, and 95% cross-seed interval summaries.

    In full mode, the 2.5% and 97.5% quantiles make robustness visible in
    figures. In fast mode, all quantiles collapse to the single seed value.
    """
    enriched = _build_seed_level_metrics(frame)
    key_columns = [
        "scenario",
        "target",
        "profile",
        "scenario_signature",
        "method",
        "confidence_level",
    ]

    q025 = lambda x: x.quantile(0.025)
    q25 = lambda x: x.quantile(0.25)
    q75 = lambda x: x.quantile(0.75)
    q975 = lambda x: x.quantile(0.975)

    summary = (
        enriched.groupby(key_columns, as_index=False)
        .agg(
            center_empirical_coverage=("empirical_coverage", "median"),
            q025_empirical_coverage=("empirical_coverage", q025),
            q25_empirical_coverage=("empirical_coverage", q25),
            q75_empirical_coverage=("empirical_coverage", q75),
            q975_empirical_coverage=("empirical_coverage", q975),

            center_relative_ci_width=("relative_ci_width", "median"),
            q025_relative_ci_width=("relative_ci_width", q025),
            q25_relative_ci_width=("relative_ci_width", q25),
            q75_relative_ci_width=("relative_ci_width", q75),
            q975_relative_ci_width=("relative_ci_width", q975),

            center_coverage_error=("coverage_error", "median"),
            q025_coverage_error=("coverage_error", q025),
            q25_coverage_error=("coverage_error", q25),
            q75_coverage_error=("coverage_error", q75),
            q975_coverage_error=("coverage_error", q975),

            n_seeds=("seed", "nunique"),
            n_replicates=("n_replicates", "first"),
            n_successful_replicates=(
                "n_successful_replicates",
                "first",
            )
            if "n_successful_replicates" in enriched.columns
            else ("n_replicates", "first"),
            n_failed_replicates=("n_failed_replicates", "first")
            if "n_failed_replicates" in enriched.columns
            else ("n_replicates", lambda x: 0),
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


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    saved: list[Path] = []
    for suffix in config.SAVE_FIGURE_FORMATS:
        path = output_dir / f"{stem}.{suffix}"
        kwargs = {"dpi": config.FIGURE_DPI} if suffix == "png" else {}
        fig.savefig(path, bbox_inches="tight", **kwargs)
        saved.append(path)
    plt.close(fig)
    return saved


def _draw_seed_interval(
    ax: plt.Axes,
    x_value: float,
    q025: float,
    q25: float,
    q75: float,
    q975: float,
    color: str,
) -> None:
    """Draw cross-seed robustness interval and IQR for one plotted point."""
    if np.isfinite(q025) and np.isfinite(q975) and q025 < q975:
        ax.vlines(
            x_value,
            q025,
            q975,
            color=color,
            linewidth=1.2,
            alpha=0.45,
            zorder=2,
        )

    if np.isfinite(q25) and np.isfinite(q75) and q25 < q75:
        ax.vlines(
            x_value,
            q25,
            q75,
            color=color,
            linewidth=3.2,
            alpha=0.55,
            zorder=2,
        )


def _adaptive_focused_ylim(
    values: list[float],
    reference_values: list[float] | None = None,
    hard_ylim: tuple[float, float] = FOCUSED_HARD_YLIM,
    padding: float = FOCUSED_Y_PADDING,
    min_span: float = FOCUSED_MIN_SPAN,
) -> tuple[float, float]:
    """Compute an adaptive focused y-axis within a hard display window.

    Only finite values inside the hard display window are used to determine
    the adaptive axis. Values outside the hard window are shown later by
    boundary triangles and do not stretch the y-axis.
    """
    hard_low, hard_high = hard_ylim

    candidate_values: list[float] = []
    for value in values:
        if np.isfinite(value) and hard_low <= value <= hard_high:
            candidate_values.append(float(value))

    if reference_values is not None:
        for value in reference_values:
            if np.isfinite(value) and hard_low <= value <= hard_high:
                candidate_values.append(float(value))

    if not candidate_values:
        return hard_ylim

    y_low = max(hard_low, min(candidate_values) - padding)
    y_high = min(hard_high, max(candidate_values) + padding)

    current_span = y_high - y_low
    if current_span < min_span:
        center = 0.5 * (y_low + y_high)
        half_span = 0.5 * min_span
        y_low = max(hard_low, center - half_span)
        y_high = min(hard_high, center + half_span)

        if y_high - y_low < min_span:
            if y_low <= hard_low:
                y_high = min(hard_high, y_low + min_span)
            elif y_high >= hard_high:
                y_low = max(hard_low, y_high - min_span)

    return y_low, y_high


def _boundary_mark_y(y_limits: tuple[float, float], side: str) -> float:
    """Return y coordinate for off-scale triangle markers."""
    y_low, y_high = y_limits
    pad = FOCUSED_MARKER_PAD_FRAC * (y_high - y_low)

    if side == "low":
        return y_low + pad
    if side == "high":
        return y_high - pad

    raise ValueError(f"Unknown side: {side}")


def _clip_value_for_axis(value: float, y_limits: tuple[float, float]) -> tuple[float, str | None]:
    """Clip a scalar value to the adaptive display axis."""
    y_low, y_high = y_limits

    if value < y_low:
        return _boundary_mark_y(y_limits, "low"), "low"
    if value > y_high:
        return _boundary_mark_y(y_limits, "high"), "high"

    return value, None


def _clip_interval_for_axis(
    q025: float,
    q25: float,
    q75: float,
    q975: float,
    y_limits: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Clip interval endpoints to the adaptive display axis."""
    y_low, y_high = y_limits

    def clip(value: float) -> float:
        if not np.isfinite(value):
            return value
        return max(y_low, min(y_high, value))

    return (clip(q025), clip(q25), clip(q75), clip(q975))


def _draw_offscale_marker(
    ax: plt.Axes,
    x_value: float,
    raw_value: float,
    y_limits: tuple[float, float],
    color: str,
) -> None:
    """Draw a boundary triangle when a point lies outside the adaptive axis."""
    y_low, y_high = y_limits

    if raw_value < y_low:
        ax.plot(
            x_value,
            _boundary_mark_y(y_limits, "low"),
            marker="v",
            linestyle="None",
            color=color,
            markersize=7.0,
            zorder=5,
        )
    elif raw_value > y_high:
        ax.plot(
            x_value,
            _boundary_mark_y(y_limits, "high"),
            marker="^",
            linestyle="None",
            color=color,
            markersize=7.0,
            zorder=5,
        )


def _draw_offscale_interval_marker(
    ax: plt.Axes,
    x_value: float,
    raw_q025: float,
    raw_q975: float,
    y_limits: tuple[float, float],
    color: str,
) -> None:
    """Draw boundary triangles when a robustness interval exceeds the axis."""
    y_low, y_high = y_limits

    if np.isfinite(raw_q025) and raw_q025 < y_low:
        ax.plot(
            x_value,
            _boundary_mark_y(y_limits, "low"),
            marker="v",
            linestyle="None",
            color=color,
            markersize=6.2,
            alpha=0.95,
            zorder=6,
        )

    if np.isfinite(raw_q975) and raw_q975 > y_high:
        ax.plot(
            x_value,
            _boundary_mark_y(y_limits, "high"),
            marker="^",
            linestyle="None",
            color=color,
            markersize=6.2,
            alpha=0.95,
            zorder=6,
        )


def _draw_classic_seed_band(
    ax: plt.Axes,
    q025: float,
    q25: float,
    q75: float,
    q975: float,
    color: str,
) -> None:
    """Draw cross-seed bands for the horizontal Classic reference line."""
    if np.isfinite(q025) and np.isfinite(q975) and q025 < q975:
        ax.axhspan(q025, q975, color=color, alpha=0.045, zorder=0)

    if np.isfinite(q25) and np.isfinite(q75) and q25 < q75:
        ax.axhspan(q25, q75, color=color, alpha=0.09, zorder=0)


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

    classic = _single_row(
        summary,
        scenario=scenario,
        target=target,
        profile="baseline",
        method="classic",
        confidence_level=confidence_level,
    )
    classic_center = float(classic["center_empirical_coverage"])
    classic_q025 = float(classic["q025_empirical_coverage"])
    classic_q25 = float(classic["q25_empirical_coverage"])
    classic_q75 = float(classic["q75_empirical_coverage"])
    classic_q975 = float(classic["q975_empirical_coverage"])
    profiled_methods = tuple(
        method
        for method in PROFILED_COVERAGE_METHODS
        if method in set(summary["method"])
    )
    offsets = _profile_offsets(profiled_methods)

    axis_values = [
        classic_center,
        classic_q025,
        classic_q25,
        classic_q75,
        classic_q975,
    ]
    for method in profiled_methods:
        for profile in profiles:
            row = _maybe_single_row(
                summary,
                scenario=scenario,
                target=target,
                profile=profile,
                method=method,
                confidence_level=confidence_level,
            )
            if row is None:
                continue
            axis_values.extend(
                [
                    float(row["center_empirical_coverage"]),
                    float(row["q025_empirical_coverage"]),
                    float(row["q25_empirical_coverage"]),
                    float(row["q75_empirical_coverage"]),
                    float(row["q975_empirical_coverage"]),
                ]
            )

    y_limits = _adaptive_focused_ylim(
        axis_values,
        reference_values=[confidence_level],
    )

    band_low, band_high = _coverage_reference_band(confidence_level, n_replicates)
    clipped_band_low = max(y_limits[0], band_low)
    clipped_band_high = min(y_limits[1], band_high)
    if clipped_band_low < clipped_band_high:
        ax.axhspan(
            clipped_band_low,
            clipped_band_high,
            color="0.85",
            alpha=0.70,
            zorder=0,
        )
    ax.axhline(
        confidence_level,
        color="0.35",
        linestyle=":",
        linewidth=1.2,
        label="_nolegend_",
    )

    ax.axhline(
        classic_center,
        color=METHOD_COLORS["classic"],
        linestyle="--",
        linewidth=1.5,
        label="_nolegend_",
    )
    classic_q025, classic_q25, classic_q75, classic_q975 = _clip_interval_for_axis(
        classic_q025,
        classic_q25,
        classic_q75,
        classic_q975,
        y_limits,
    )
    _draw_classic_seed_band(
        ax,
        classic_q025,
        classic_q25,
        classic_q75,
        classic_q975,
        METHOD_COLORS["classic"],
    )

    for method in profiled_methods:
        x_values: list[float] = []
        raw_y_values: list[float] = []
        raw_q025_values: list[float] = []
        raw_q975_values: list[float] = []
        y_values: list[float] = []
        q025_values: list[float] = []
        q25_values: list[float] = []
        q75_values: list[float] = []
        q975_values: list[float] = []

        for profile_index, profile in enumerate(profiles):
            row = _maybe_single_row(
                summary,
                scenario=scenario,
                target=target,
                profile=profile,
                method=method,
                confidence_level=confidence_level,
            )
            if row is None:
                continue
            raw_center = float(row["center_empirical_coverage"])
            raw_q025 = float(row["q025_empirical_coverage"])
            raw_q25 = float(row["q25_empirical_coverage"])
            raw_q75 = float(row["q75_empirical_coverage"])
            raw_q975 = float(row["q975_empirical_coverage"])

            plotted_center, _ = _clip_value_for_axis(raw_center, y_limits)

            q025, q25, q75, q975 = _clip_interval_for_axis(
                raw_q025,
                raw_q25,
                raw_q75,
                raw_q975,
                y_limits,
            )

            x_values.append(x_base[profile_index] + offsets[method])
            raw_y_values.append(raw_center)
            raw_q025_values.append(raw_q025)
            raw_q975_values.append(raw_q975)
            y_values.append(plotted_center)
            q025_values.append(q025)
            q25_values.append(q25)
            q75_values.append(q75)
            q975_values.append(q975)

        if not x_values:
            continue

        method_color = METHOD_COLORS[method]
        ax.plot(
            x_values,
            y_values,
            marker=CALIBRATION_MARKERS[method],
            linestyle="None",
            markersize=6.5,
            alpha=0.76 if method == "naive_ml" else 0.96,
            color=method_color,
            label="_nolegend_",
            zorder=3,
        )

        for x_value, q025, q25, q75, q975 in zip(
            x_values, q025_values, q25_values, q75_values, q975_values
        ):
            _draw_seed_interval(ax, x_value, q025, q25, q75, q975, method_color)

        for x_value, raw_q025, raw_q975 in zip(
            x_values, raw_q025_values, raw_q975_values
        ):
            _draw_offscale_interval_marker(
                ax,
                x_value,
                raw_q025,
                raw_q975,
                y_limits,
                method_color,
            )

        for x_value, raw_y in zip(x_values, raw_y_values):
            _draw_offscale_marker(ax, x_value, raw_y, y_limits, method_color)

    ax.set_xticks(x_base)
    ax.set_xticklabels([PROFILE_LABELS[profile] for profile in profiles])
    ax.set_ylim(*y_limits)
    ax.set_ylabel("Empirical coverage\n(adaptive focused axis)")
    ax.text(
        0.01,
        0.03,
        "▲/▼ outside display range",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="0.30",
    )
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
    efficiency_methods = tuple(
        method
        for method in EFFICIENCY_METHODS
        if method in set(summary["method"])
    )
    offsets = _profile_offsets(efficiency_methods)

    axis_values = [1.0]
    for method in efficiency_methods:
        for profile in profiles:
            row = _maybe_single_row(
                summary,
                scenario=scenario,
                target=target,
                profile=profile,
                method=method,
                confidence_level=confidence_level,
            )
            if row is None:
                continue
            axis_values.extend(
                [
                    float(row["center_relative_ci_width"]),
                    float(row["q025_relative_ci_width"]),
                    float(row["q25_relative_ci_width"]),
                    float(row["q75_relative_ci_width"]),
                    float(row["q975_relative_ci_width"]),
                ]
            )

    y_limits = _adaptive_focused_ylim(
        axis_values,
        reference_values=[1.0],
    )

    ax.axhline(
        1.0,
        color=METHOD_COLORS["classic"],
        linestyle="--",
        linewidth=1.5,
        label="_nolegend_",
    )
    ax.text(
        0.98,
        1.0,
        "Classic = 1",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8.5,
    )

    for method in efficiency_methods:
        x_values: list[float] = []
        raw_y_values: list[float] = []
        raw_q025_values: list[float] = []
        raw_q975_values: list[float] = []
        y_values: list[float] = []
        q025_values: list[float] = []
        q25_values: list[float] = []
        q75_values: list[float] = []
        q975_values: list[float] = []

        for profile_index, profile in enumerate(profiles):
            row = _maybe_single_row(
                summary,
                scenario=scenario,
                target=target,
                profile=profile,
                method=method,
                confidence_level=confidence_level,
            )
            if row is None:
                continue
            raw_center = float(row["center_relative_ci_width"])
            raw_q025 = float(row["q025_relative_ci_width"])
            raw_q25 = float(row["q25_relative_ci_width"])
            raw_q75 = float(row["q75_relative_ci_width"])
            raw_q975 = float(row["q975_relative_ci_width"])

            plotted_center, _ = _clip_value_for_axis(raw_center, y_limits)

            q025, q25, q75, q975 = _clip_interval_for_axis(
                raw_q025,
                raw_q25,
                raw_q75,
                raw_q975,
                y_limits,
            )

            x_values.append(x_base[profile_index] + offsets[method])
            raw_y_values.append(raw_center)
            raw_q025_values.append(raw_q025)
            raw_q975_values.append(raw_q975)
            y_values.append(plotted_center)
            q025_values.append(q025)
            q25_values.append(q25)
            q75_values.append(q75)
            q975_values.append(q975)

        if not x_values:
            continue

        method_color = METHOD_COLORS[method]
        ax.plot(
            x_values,
            y_values,
            marker=CALIBRATION_MARKERS.get(method, "o"),
            linestyle="None",
            markersize=6.5,
            color=method_color,
            label="_nolegend_",
            zorder=3,
        )

        for x_value, q025, q25, q75, q975 in zip(
            x_values, q025_values, q25_values, q75_values, q975_values
        ):
            _draw_seed_interval(ax, x_value, q025, q25, q75, q975, method_color)

        # Mark off-scale robustness intervals, even when the median point is inside.
        for x_value, raw_q025, raw_q975 in zip(
            x_values, raw_q025_values, raw_q975_values
        ):
            _draw_offscale_interval_marker(
                ax,
                x_value,
                raw_q025,
                raw_q975,
                y_limits,
                method_color,
            )

        # Mark off-scale median points.
        for x_value, raw_y in zip(x_values, raw_y_values):
            _draw_offscale_marker(ax, x_value, raw_y, y_limits, method_color)

    ax.set_xticks(x_base)
    ax.set_xticklabels([PROFILE_LABELS[profile] for profile in profiles])
    ax.set_ylim(*y_limits)
    ax.set_ylabel("Relative CI width\n(adaptive focused axis)")
    ax.grid(axis="y", alpha=0.25)


def _plot_main_figure(
    summary: pd.DataFrame,
    output_dir: Path,
    scenario_name: str,
    confidence_level: float,
) -> list[Path]:
    scenario = get_scenario(scenario_name)
    row_count = len(scenario.target_names)
    fig, axes = plt.subplots(
        row_count,
        2,
        figsize=(12.5, max(5.2, 3.85 * row_count + 0.8)),
        squeeze=False,
    )
    confidence_text = f"{confidence_level * 100:.1f}%"
    fig.suptitle(
        f"{display_name(scenario_name)}: inference validity and efficiency ({confidence_text} confidence)",
        fontsize=12,
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

    legend_methods = (
        "classic",
        "naive_ml",
        "ppi",
        "ppi_plus_plus_v1",
        "ppi_plus_plus_v2",
    )
    legend_handles = _method_legend_handles(legend_methods)
    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        ncol=min(5, len(legend_handles)),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        frameon=False,
        columnspacing=1.25,
        handletextpad=0.6,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.78 if row_count == 1 else 0.84,
        bottom=0.10,
        wspace=0.18,
        hspace=0.34,
    )
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
    """Draw coverage error by nominal level with cross-seed robustness intervals.

    The shared y-axis is deliberately focused on plausible calibration error.
    A near-zero-width Naive-ML interval can produce coverage error near -1;
    showing that value on the same linear scale would conceal whether PPI-family
    methods are calibrated.  Such points are therefore marked with a downward
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
    ax.axhline(
        0.0,
        color="0.35",
        linestyle=":",
        linewidth=1.1,
        label="_nolegend_",
    )

    for method in COVERAGE_METHODS:
        centers: list[float] = []
        q025s: list[float] = []
        q25s: list[float] = []
        q75s: list[float] = []
        q975s: list[float] = []
        x_values: list[float] = []

        for level in levels:
            source_profile = "baseline" if method == "classic" else profile
            row = _maybe_single_row(
                summary,
                scenario=scenario,
                target=target,
                profile=source_profile,
                method=method,
                confidence_level=float(level),
            )
            if row is None:
                continue
            x_values.append(float(level))
            centers.append(float(row["center_coverage_error"]))
            q025s.append(float(row["q025_coverage_error"]))
            q25s.append(float(row["q25_coverage_error"]))
            q75s.append(float(row["q75_coverage_error"]))
            q975s.append(float(row["q975_coverage_error"]))

        if not x_values:
            continue

        x_array = np.asarray(x_values, dtype=float)
        center_array = np.asarray(centers, dtype=float)
        q025_array = np.asarray(q025s, dtype=float)
        q25_array = np.asarray(q25s, dtype=float)
        q75_array = np.asarray(q75s, dtype=float)
        q975_array = np.asarray(q975s, dtype=float)

        within = (center_array >= lower_limit) & (center_array <= upper_limit)
        lower_censored = center_array < lower_limit
        upper_censored = center_array > upper_limit

        style = "--" if method == "classic" else "-"
        alpha = 0.75 if method == "naive_ml" else 0.96
        color = METHOD_COLORS[method]

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

            for x_value, q025, q25, q75, q975 in zip(
                x_array[within],
                q025_array[within],
                q25_array[within],
                q75_array[within],
                q975_array[within],
            ):
                _draw_seed_interval(
                    ax,
                    float(x_value),
                    max(lower_limit, float(q025)),
                    max(lower_limit, float(q25)),
                    min(upper_limit, float(q75)),
                    min(upper_limit, float(q975)),
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
) -> list[Path]:
    scenario = get_scenario(scenario_name)
    profiles = list(config.ACTIVE_PROFILES)
    target_count = len(scenario.target_names)
    y_limits = tuple(float(value) for value in config.CALIBRATION_COVERAGE_ERROR_YLIM)
    fig, axes = plt.subplots(
        target_count,
        len(profiles),
        figsize=(max(14.0, 3.5 * len(profiles)), max(8.0, 2.65 * target_count)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    fig.suptitle(
        f"{display_name(scenario_name)}: coverage calibration across nominal confidence levels",
        fontsize=12,
        y=0.985,
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

    legend_handles = _calibration_legend_handles()
    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        ncol=min(6, len(legend_handles)),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        frameon=False,
        columnspacing=1.25,
        handletextpad=0.6,
    )
    fig.text(
        0.5,
        0.018,
        "Coverage error = empirical coverage - nominal. ▲/▼ mark values outside the displayed range.",
        ha="center",
        va="bottom",
        fontsize=8.0,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.86,
        bottom=0.105,
        wspace=0.16,
        hspace=0.26,
    )
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
    labelled_summary = summary.copy()
    labelled_summary["method_label"] = labelled_summary["method"].map(METHOD_LABELS)
    summary_columns = list(summary.columns)
    method_index = summary_columns.index("method")
    ordered_columns = (
        summary_columns[: method_index + 1]
        + ["method_label"]
        + summary_columns[method_index + 1 :]
    )
    labelled_summary[ordered_columns].to_csv(destination, index=False)
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

    main_level = float(config.MAIN_FIGURE_CONFIDENCE_LEVEL)
    if main_level not in {float(level) for level in config.CONFIDENCE_LEVELS}:
        raise ValueError("MAIN_FIGURE_CONFIDENCE_LEVEL must be present in CONFIDENCE_LEVELS.")

    saved: list[Path] = []
    for scenario_name in config.ACTIVE_SCENARIOS:
        saved.extend(_plot_main_figure(summary, output_dir, scenario_name, main_level))
        saved.extend(_plot_calibration_figure(summary, output_dir, scenario_name))
    return saved
