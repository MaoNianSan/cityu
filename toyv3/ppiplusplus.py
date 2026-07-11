"""PPI++ estimators.

V1 delegates to the official ppi-python package with lam=None. V2 keeps the
project's transparent lambda-grid search over the internal weighted PPI solver.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import norm

import config
from data_generation import SimulationData
from formulation import EstimatorResult, ScenarioSpec, interval_key
from learner_proxy import PredictionBundle
from ppi import fit_weighted_ppi

try:
    from ppi_py import (
        ppi_logistic_ci,
        ppi_logistic_pointestimate,
        ppi_mean_ci,
        ppi_mean_pointestimate,
        ppi_ols_ci,
        ppi_ols_pointestimate,
    )
except Exception as exc:  # pragma: no cover - exercised only without dependency
    ppi_mean_pointestimate = None
    ppi_mean_ci = None
    ppi_ols_pointestimate = None
    ppi_ols_ci = None
    ppi_logistic_pointestimate = None
    ppi_logistic_ci = None
    _PPI_PY_IMPORT_EXCEPTION: Exception | None = exc
else:
    _PPI_PY_IMPORT_EXCEPTION = None

Array = np.ndarray


def _failed_result(
    method: str,
    p: int,
    reason: str,
    diagnostics: dict | None = None,
) -> EstimatorResult:
    payload = dict(diagnostics or {})
    payload["failure_reason"] = reason
    return EstimatorResult(
        method=method,
        estimate=np.full(p, np.nan),
        covariance=np.full((p, p), np.nan),
        converged=False,
        diagnostics=payload,
    )


def _candidate_lambdas_v2() -> tuple[float, ...]:
    if config.PPI_PLUS_PLUS_V2_LAMBDA_MODE == "fixed":
        if config.PPI_PLUS_PLUS_V2_FIXED_LAMBDA is None:
            raise ValueError(
                "PPI_PLUS_PLUS_V2_FIXED_LAMBDA must be set when lambda mode is 'fixed'."
            )
        value = float(config.PPI_PLUS_PLUS_V2_FIXED_LAMBDA)
        if not 0.0 <= value <= 1.0:
            raise ValueError("PPI_PLUS_PLUS_V2_FIXED_LAMBDA must lie in [0, 1].")
        return (value,)
    if config.PPI_PLUS_PLUS_V2_LAMBDA_MODE == "min_sandwich_trace":
        return tuple(float(value) for value in config.PPI_PLUS_PLUS_V2_LAMBDA_GRID)
    raise ValueError(
        f"Unknown lambda mode {config.PPI_PLUS_PLUS_V2_LAMBDA_MODE!r}."
    )


def _as_target_vector(value: object, p: int) -> Array:
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = array.reshape(1)
    array = array.reshape(-1)
    if array.shape != (p,):
        raise ValueError(f"Expected estimate shape {(p,)}, got {array.shape}.")
    return array


def _validate_ci_bounds(lower: object, upper: object, p: int) -> tuple[Array, Array]:
    lower_array = _as_target_vector(lower, p)
    upper_array = _as_target_vector(upper, p)
    if not (np.all(np.isfinite(lower_array)) and np.all(np.isfinite(upper_array))):
        raise ValueError("ppi-python returned non-finite confidence interval bounds.")
    if np.any(upper_array < lower_array):
        raise ValueError("ppi-python returned an upper CI bound below its lower bound.")
    return lower_array, upper_array


def _diagonal_covariance_from_ci(lower: Array, upper: Array) -> Array:
    confidence_level = float(config.PPI_PLUS_PLUS_V1_COVARIANCE_CONFIDENCE_LEVEL)
    z_value = float(norm.ppf(0.5 + confidence_level / 2.0))
    if not np.isfinite(z_value) or z_value <= 0.0:
        raise ValueError("Invalid PPI++V1 covariance confidence level.")
    standard_errors = (upper - lower) / (2.0 * z_value)
    covariance = np.diag(standard_errors**2)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Backed-out diagonal covariance is non-finite.")
    return covariance


def _ppi_python_functions(
    scenario: ScenarioSpec,
) -> tuple[Callable[..., object], Callable[..., tuple[object, object]]]:
    if scenario.family == "mean":
        if ppi_mean_pointestimate is None or ppi_mean_ci is None:
            raise ImportError("ppi-python is not available.") from _PPI_PY_IMPORT_EXCEPTION
        return ppi_mean_pointestimate, ppi_mean_ci
    if scenario.family == "linear":
        if ppi_ols_pointestimate is None or ppi_ols_ci is None:
            raise ImportError("ppi-python is not available.") from _PPI_PY_IMPORT_EXCEPTION
        return ppi_ols_pointestimate, ppi_ols_ci
    if scenario.family == "logistic":
        if ppi_logistic_pointestimate is None or ppi_logistic_ci is None:
            raise ImportError("ppi-python is not available.") from _PPI_PY_IMPORT_EXCEPTION
        return ppi_logistic_pointestimate, ppi_logistic_ci
    raise ValueError(f"Unsupported family {scenario.family!r}.")


def _call_ppi_python_pointestimate(
    scenario: ScenarioSpec,
    pointestimate_fn: Callable[..., object],
    data: SimulationData,
    prediction: PredictionBundle,
) -> object:
    y_labeled = data.outcome_for(scenario)
    if scenario.family == "mean":
        return pointestimate_fn(
            y_labeled,
            prediction.f_labeled,
            prediction.f_unlabeled,
            lam=None,
            coord=None,
            lam_optim_mode="overall",
        )
    optimizer_options = (
        {"maxiter": config.GLM_MAX_ITER} if scenario.family == "logistic" else None
    )
    common_args = (
        data.x_labeled,
        y_labeled,
        prediction.f_labeled,
        data.x_unlabeled,
        prediction.f_unlabeled,
    )
    if scenario.family == "logistic":
        return pointestimate_fn(
            *common_args,
            lam=None,
            coord=None,
            optimizer_options=optimizer_options,
        )
    return pointestimate_fn(*common_args, lam=None, coord=None)


def _call_ppi_python_ci(
    scenario: ScenarioSpec,
    ci_fn: Callable[..., tuple[object, object]],
    data: SimulationData,
    prediction: PredictionBundle,
    confidence_level: float,
) -> tuple[object, object]:
    y_labeled = data.outcome_for(scenario)
    alpha = 1.0 - float(confidence_level)
    if scenario.family == "mean":
        return ci_fn(
            y_labeled,
            prediction.f_labeled,
            prediction.f_unlabeled,
            alpha=alpha,
            alternative="two-sided",
            lam=None,
            coord=None,
            lam_optim_mode="overall",
        )
    optimizer_options = (
        {"maxiter": config.GLM_MAX_ITER} if scenario.family == "logistic" else None
    )
    common_args = (
        data.x_labeled,
        y_labeled,
        prediction.f_labeled,
        data.x_unlabeled,
        prediction.f_unlabeled,
    )
    if scenario.family == "logistic":
        return ci_fn(
            *common_args,
            alpha=alpha,
            alternative="two-sided",
            lam=None,
            coord=None,
            optimizer_options=optimizer_options,
        )
    return ci_fn(
        *common_args,
        alpha=alpha,
        alternative="two-sided",
        lam=None,
        coord=None,
    )


def fit_ppi_plus_plus_v1(
    scenario: ScenarioSpec,
    data: SimulationData,
    prediction: PredictionBundle,
) -> EstimatorResult:
    """Fit package-standard PPI++ via ppi-python with ``lam=None``.

    Every configured marginal interval is requested directly from ppi-python.
    A 95% diagonal reconstruction remains only for interface compatibility.
    """
    method = "ppi_plus_plus_v1"
    p = len(scenario.target_names)
    diagnostics = {
        "lambda_selection_mode": "ppi_python_internal_lam_none",
        "lambda_value_available": False,
        "lambda_recomputed_per_ci_call": True,
        "interval_source": "ppi_python_direct",
        "covariance_source": "diagonal_reconstruction_for_compatibility_only",
        "covariance_confidence_level": float(
            config.PPI_PLUS_PLUS_V1_COVARIANCE_CONFIDENCE_LEVEL
        ),
    }
    try:
        pointestimate_fn, ci_fn = _ppi_python_functions(scenario)
        estimate = _as_target_vector(
            _call_ppi_python_pointestimate(scenario, pointestimate_fn, data, prediction),
            p,
        )
        intervals = {}
        calls = []
        for level in config.CONFIDENCE_LEVELS:
            lower, upper = _validate_ci_bounds(
                *_call_ppi_python_ci(scenario, ci_fn, data, prediction, level), p
            )
            intervals[interval_key(level)] = (lower, upper)
            calls.append({"confidence_level": float(level), "alpha": 1.0 - float(level), "lam": None})
        compatibility = intervals[interval_key(config.PPI_PLUS_PLUS_V1_COVARIANCE_CONFIDENCE_LEVEL)]
        covariance = _diagonal_covariance_from_ci(*compatibility)
        diagnostics["interval_calls"] = calls
        if not (np.all(np.isfinite(estimate)) and np.all(np.isfinite(covariance))):
            raise ValueError("ppi-python returned non-finite estimate or covariance.")
    except Exception as exc:
        diagnostics["exception"] = repr(exc)
        return _failed_result(method, p, "ppi_python_call_failed", diagnostics)

    return EstimatorResult(
        method=method,
        estimate=estimate,
        covariance=covariance,
        diagnostics=diagnostics,
        intervals=intervals,
    )


def fit_ppi_plus_plus_v2(
    scenario: ScenarioSpec,
    data: SimulationData,
    prediction: PredictionBundle,
) -> EstimatorResult:
    """Select one scalar lambda by minimizing plug-in sandwich covariance trace."""
    method = "ppi_plus_plus_v2"
    best_result: EstimatorResult | None = None
    best_trace = np.inf
    previous_beta = None
    traces: dict[str, float | None] = {}

    for lambda_ in _candidate_lambdas_v2():
        result = fit_weighted_ppi(
            scenario=scenario,
            data=data,
            prediction=prediction,
            lambda_=lambda_,
            method=method,
            initial_beta=previous_beta,
        )
        trace_key = f"{lambda_:.3f}"
        if result.converged and np.all(np.isfinite(result.covariance)):
            trace = float(np.trace(result.covariance))
            traces[trace_key] = trace
            if scenario.family == "logistic":
                previous_beta = result.estimate.copy()
            # Strict inequality preserves the smaller lambda under a tie.
            if trace < best_trace - 1e-15:
                best_trace = trace
                best_result = result
        else:
            traces[trace_key] = None

    if best_result is None:
        return _failed_result(
            method,
            len(scenario.target_names),
            "all_lambda_candidates_failed",
            {
                "lambda_selection_mode": config.PPI_PLUS_PLUS_V2_LAMBDA_MODE,
                "lambda_candidate_traces": traces,
            },
        )

    best_result.method = method
    best_result.diagnostics.update(
        {
            "lambda_selection_mode": config.PPI_PLUS_PLUS_V2_LAMBDA_MODE,
            "selected_lambda": best_result.diagnostics.get("lambda_hat"),
            "selected_covariance_trace": best_trace,
            "lambda_candidate_traces": traces,
        }
    )
    return best_result


def fit_ppi_plus_plus(
    scenario: ScenarioSpec,
    data: SimulationData,
    prediction: PredictionBundle,
) -> EstimatorResult:
    """Backward-compatible alias for the internal grid-search PPI++ implementation."""
    return fit_ppi_plus_plus_v2(scenario, data, prediction)
