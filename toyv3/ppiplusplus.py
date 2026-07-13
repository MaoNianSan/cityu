"""PPI++ estimators.

V1 follows ``ppi-python==0.2.3`` and makes the package's internally selected
lambda explicit.  The package's public ``ci(..., lam=None)`` path performs a
second plug-in lambda update for OLS and logistic regression.  We reproduce
that path once, then pass the resulting scalar lambda explicitly to the point
estimate and every confidence-interval call.  This guarantees that the stored
point estimate, all interval midpoints, and the reported lambda refer to the
same fitted estimator.

V2 keeps the project's transparent lambda-grid search over the internal
weighted-PPI solver.
"""
from __future__ import annotations

from collections.abc import Callable
from importlib import metadata

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
    from ppi_py.ppi import (
        _calc_lam_glm,
        _logistic_get_stats,
        _ols_get_stats,
        construct_weight_vector,
        reshape_to_2d,
    )
except Exception as exc:  # pragma: no cover - exercised only without dependency
    ppi_mean_pointestimate = None
    ppi_mean_ci = None
    ppi_ols_pointestimate = None
    ppi_ols_ci = None
    ppi_logistic_pointestimate = None
    ppi_logistic_ci = None
    _calc_lam_glm = None
    _ols_get_stats = None
    _logistic_get_stats = None
    construct_weight_vector = None
    reshape_to_2d = None
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


def _ppi_python_version() -> str:
    try:
        return metadata.version("ppi-python")
    except metadata.PackageNotFoundError:
        return "unavailable"


def _require_supported_ppi_python() -> None:
    if _PPI_PY_IMPORT_EXCEPTION is not None:
        raise ImportError("ppi-python is not available.") from _PPI_PY_IMPORT_EXCEPTION
    version = _ppi_python_version()
    expected = str(config.PPI_PY_EXPECTED_VERSION)
    if version != expected:
        raise RuntimeError(
            "PPI++V1 uses private ppi-python helper functions to reproduce the "
            f"package CI path. Expected ppi-python=={expected}, found {version}."
        )


def _ppi_python_functions(
    scenario: ScenarioSpec,
) -> tuple[Callable[..., object], Callable[..., tuple[object, object]]]:
    _require_supported_ppi_python()
    if scenario.family == "mean":
        assert ppi_mean_pointestimate is not None and ppi_mean_ci is not None
        return ppi_mean_pointestimate, ppi_mean_ci
    if scenario.family == "linear":
        assert ppi_ols_pointestimate is not None and ppi_ols_ci is not None
        return ppi_ols_pointestimate, ppi_ols_ci
    if scenario.family == "logistic":
        assert ppi_logistic_pointestimate is not None and ppi_logistic_ci is not None
        return ppi_logistic_pointestimate, ppi_logistic_ci
    raise ValueError(f"Unsupported family {scenario.family!r}.")


def _common_regression_args(
    scenario: ScenarioSpec,
    data: SimulationData,
    prediction: PredictionBundle,
) -> tuple[Array, Array, Array, Array, Array]:
    return (
        np.asarray(data.x_labeled, dtype=float),
        np.asarray(data.outcome_for(scenario), dtype=float),
        np.asarray(prediction.f_labeled, dtype=float),
        np.asarray(data.x_unlabeled, dtype=float),
        np.asarray(prediction.f_unlabeled, dtype=float),
    )


def _estimate_v1_lambda(
    scenario: ScenarioSpec,
    data: SimulationData,
    prediction: PredictionBundle,
) -> tuple[float, float | None]:
    """Return the final lambda used by ``ppi-python``'s ``ci(lam=None)`` path.

    Returns
    -------
    final_lambda, stage1_lambda
        For mean estimation the two values are identical.  For OLS and
        logistic regression, ``stage1_lambda`` is the lambda selected by the
        package point-estimate path; ``final_lambda`` is the second plug-in
        update used by the package CI path.
    """
    _require_supported_ppi_python()
    assert _calc_lam_glm is not None

    y_labeled = np.asarray(data.outcome_for(scenario), dtype=float)
    f_labeled = np.asarray(prediction.f_labeled, dtype=float)
    f_unlabeled = np.asarray(prediction.f_unlabeled, dtype=float)

    if scenario.family == "mean":
        assert reshape_to_2d is not None and construct_weight_vector is not None
        Y = reshape_to_2d(y_labeled)
        Yhat = reshape_to_2d(f_labeled)
        Yhat_unlabeled = reshape_to_2d(f_unlabeled)
        n = Y.shape[0]
        N = Yhat_unlabeled.shape[0]
        d = Yhat.shape[1]
        w = construct_weight_vector(n, None, vectorized=True)
        w_unlabeled = construct_weight_vector(N, None, vectorized=True)
        # This is the pilot used by ppi_mean_ci(..., lam=None).
        pilot = np.asarray(
            ppi_mean_pointestimate(
                Y,
                Yhat,
                Yhat_unlabeled,
                lam=1.0,
                w=w,
                w_unlabeled=w_unlabeled,
            ),
            dtype=float,
        ).reshape(1, -1)
        grads = w * (Y - pilot)
        grads_hat = w * (Yhat - pilot)
        grads_hat_unlabeled = w_unlabeled * (Yhat_unlabeled - pilot)
        lam = float(
            _calc_lam_glm(
                grads,
                grads_hat,
                grads_hat_unlabeled,
                np.eye(d),
                coord=None,
                clip=True,
                optim_mode="overall",
            )
        )
        return lam, lam

    pointestimate_fn, _ = _ppi_python_functions(scenario)
    common_args = _common_regression_args(scenario, data, prediction)
    optimizer_options = (
        {"maxiter": config.GLM_MAX_ITER} if scenario.family == "logistic" else None
    )
    if scenario.family == "logistic":
        stage1_estimate = _as_target_vector(
            pointestimate_fn(
                *common_args,
                lam=None,
                coord=None,
                optimizer_options=optimizer_options,
            ),
            len(scenario.target_names),
        )
    else:
        stage1_estimate = _as_target_vector(
            pointestimate_fn(*common_args, lam=None, coord=None),
            len(scenario.target_names),
        )

    X, Y, Yhat, X_unlabeled, Yhat_unlabeled = common_args
    n = Y.shape[0]
    N = Yhat_unlabeled.shape[0]
    w = np.ones(n, dtype=float)
    w_unlabeled = np.ones(N, dtype=float)

    # Recover the first-stage lambda for auditability.  It is the lambda that
    # maps the package's lam=1 pilot to ``stage1_estimate``.
    if scenario.family == "linear":
        assert _ols_get_stats is not None
        pilot_lam1 = _as_target_vector(
            ppi_ols_pointestimate(
                X, Y, Yhat, X_unlabeled, Yhat_unlabeled, lam=1.0, coord=None
            ),
            len(scenario.target_names),
        )
        stats1 = _ols_get_stats(
            pilot_lam1,
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            w=w,
            w_unlabeled=w_unlabeled,
            use_unlabeled=True,
        )
        stage1_lambda = float(_calc_lam_glm(*stats1, coord=None, clip=True))
        stats2 = _ols_get_stats(
            stage1_estimate,
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            w=w,
            w_unlabeled=w_unlabeled,
            use_unlabeled=True,
        )
    else:
        assert _logistic_get_stats is not None
        pilot_lam1 = _as_target_vector(
            ppi_logistic_pointestimate(
                X,
                Y,
                Yhat,
                X_unlabeled,
                Yhat_unlabeled,
                lam=1.0,
                coord=None,
                optimizer_options=optimizer_options,
            ),
            len(scenario.target_names),
        )
        stats1 = _logistic_get_stats(
            pilot_lam1,
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            w=w,
            w_unlabeled=w_unlabeled,
            use_unlabeled=True,
        )
        stage1_lambda = float(_calc_lam_glm(*stats1, coord=None, clip=True))
        stats2 = _logistic_get_stats(
            stage1_estimate,
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            w=w,
            w_unlabeled=w_unlabeled,
            use_unlabeled=True,
        )

    final_lambda = float(_calc_lam_glm(*stats2, coord=None, clip=True))
    return final_lambda, stage1_lambda


def _call_ppi_python_pointestimate(
    scenario: ScenarioSpec,
    pointestimate_fn: Callable[..., object],
    data: SimulationData,
    prediction: PredictionBundle,
    lambda_hat: float,
) -> object:
    y_labeled = data.outcome_for(scenario)
    if scenario.family == "mean":
        return pointestimate_fn(
            y_labeled,
            prediction.f_labeled,
            prediction.f_unlabeled,
            lam=lambda_hat,
            coord=None,
            lam_optim_mode="overall",
        )
    common_args = _common_regression_args(scenario, data, prediction)
    if scenario.family == "logistic":
        return pointestimate_fn(
            *common_args,
            lam=lambda_hat,
            coord=None,
            optimizer_options={"maxiter": config.GLM_MAX_ITER},
        )
    return pointestimate_fn(*common_args, lam=lambda_hat, coord=None)


def _call_ppi_python_ci(
    scenario: ScenarioSpec,
    ci_fn: Callable[..., tuple[object, object]],
    data: SimulationData,
    prediction: PredictionBundle,
    confidence_level: float,
    lambda_hat: float,
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
            lam=lambda_hat,
            coord=None,
            lam_optim_mode="overall",
        )
    common_args = _common_regression_args(scenario, data, prediction)
    if scenario.family == "logistic":
        return ci_fn(
            *common_args,
            alpha=alpha,
            alternative="two-sided",
            lam=lambda_hat,
            coord=None,
            optimizer_options={"maxiter": config.GLM_MAX_ITER},
        )
    return ci_fn(
        *common_args,
        alpha=alpha,
        alternative="two-sided",
        lam=lambda_hat,
        coord=None,
    )


def fit_ppi_plus_plus_v1(
    scenario: ScenarioSpec,
    data: SimulationData,
    prediction: PredictionBundle,
) -> EstimatorResult:
    """Fit package-standard PPI++ with one coherent, auditable lambda.

    The final lambda reproduces the estimator centered by
    ``ppi-python==0.2.3``'s direct ``ci(..., lam=None)`` path.  It is estimated
    once per replicate/profile and then supplied explicitly to the point
    estimate and all configured confidence intervals.
    """
    method = "ppi_plus_plus_v1"
    p = len(scenario.target_names)
    diagnostics = {
        "lambda_selection_mode": "ppi_python_0.2.3_ci_equivalent_two_stage",
        "lambda_value_available": True,
        "lambda_recomputed_per_ci_call": False,
        "interval_source": "ppi_python_fixed_selected_lambda",
        "covariance_source": "diagonal_reconstruction_for_compatibility_only",
        "covariance_confidence_level": float(
            config.PPI_PLUS_PLUS_V1_COVARIANCE_CONFIDENCE_LEVEL
        ),
        "ppi_python_version": _ppi_python_version(),
    }
    try:
        pointestimate_fn, ci_fn = _ppi_python_functions(scenario)
        lambda_hat, stage1_lambda = _estimate_v1_lambda(scenario, data, prediction)
        estimate = _as_target_vector(
            _call_ppi_python_pointestimate(
                scenario,
                pointestimate_fn,
                data,
                prediction,
                lambda_hat,
            ),
            p,
        )
        intervals: dict[float, tuple[Array, Array]] = {}
        calls = []
        for level in config.CONFIDENCE_LEVELS:
            lower, upper = _validate_ci_bounds(
                *_call_ppi_python_ci(
                    scenario,
                    ci_fn,
                    data,
                    prediction,
                    level,
                    lambda_hat,
                ),
                p,
            )
            midpoint = 0.5 * (lower + upper)
            if not np.allclose(midpoint, estimate, atol=1e-8, rtol=1e-8):
                raise ValueError(
                    "PPI++V1 interval midpoint does not match the stored point estimate."
                )
            intervals[interval_key(level)] = (lower, upper)
            calls.append(
                {
                    "confidence_level": float(level),
                    "alpha": 1.0 - float(level),
                    "lam": lambda_hat,
                }
            )
        compatibility = intervals[
            interval_key(config.PPI_PLUS_PLUS_V1_COVARIANCE_CONFIDENCE_LEVEL)
        ]
        covariance = _diagonal_covariance_from_ci(*compatibility)
        diagnostics.update(
            {
                "lambda_hat": lambda_hat,
                "selected_lambda": lambda_hat,
                "stage1_lambda": stage1_lambda,
                "interval_calls": calls,
            }
        )
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
