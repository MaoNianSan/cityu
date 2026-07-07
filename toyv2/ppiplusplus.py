"""PPI++ with a configurable scalar lambda-selection interface."""
from __future__ import annotations

import numpy as np

import config
from data_generation import SimulationData
from formulation import EstimatorResult, ScenarioSpec
from learner_proxy import PredictionBundle
from ppi import fit_weighted_ppi


def _candidate_lambdas() -> tuple[float, ...]:
    if config.PPI_PLUS_PLUS_LAMBDA_MODE == "fixed":
        if config.PPI_PLUS_PLUS_FIXED_LAMBDA is None:
            raise ValueError("PPI_PLUS_PLUS_FIXED_LAMBDA must be set when lambda mode is 'fixed'.")
        value = float(config.PPI_PLUS_PLUS_FIXED_LAMBDA)
        if not 0.0 <= value <= 1.0:
            raise ValueError("PPI_PLUS_PLUS_FIXED_LAMBDA must lie in [0, 1].")
        return (value,)
    if config.PPI_PLUS_PLUS_LAMBDA_MODE == "min_sandwich_trace":
        return tuple(float(value) for value in config.PPI_PLUS_PLUS_LAMBDA_GRID)
    raise ValueError(f"Unknown lambda mode {config.PPI_PLUS_PLUS_LAMBDA_MODE!r}.")


def fit_ppi_plus_plus(
    scenario: ScenarioSpec,
    data: SimulationData,
    prediction: PredictionBundle,
) -> EstimatorResult:
    """Select one scalar lambda and fit PPI++.

    The selected lambda minimizes the trace of the plug-in sandwich covariance.
    For the mean scenario this trace is simply the estimated scalar variance.
    """
    best_result: EstimatorResult | None = None
    best_trace = np.inf
    previous_beta = None
    traces: dict[str, float | None] = {}

    for lambda_ in _candidate_lambdas():
        result = fit_weighted_ppi(
            scenario=scenario,
            data=data,
            prediction=prediction,
            lambda_=lambda_,
            method="ppi_plus_plus",
            initial_beta=previous_beta,
        )
        if result.converged and np.all(np.isfinite(result.covariance)):
            trace = float(np.trace(result.covariance))
            traces[f"{lambda_:.2f}"] = trace
            if scenario.family == "logistic":
                previous_beta = result.estimate.copy()
            # Strict inequality preserves the smaller lambda under a tie.
            if trace < best_trace - 1e-15:
                best_trace = trace
                best_result = result
        else:
            traces[f"{lambda_:.2f}"] = None

    if best_result is None:
        return EstimatorResult(
            method="ppi_plus_plus",
            estimate=np.full(len(scenario.target_names), np.nan),
            covariance=np.full((len(scenario.target_names), len(scenario.target_names)), np.nan),
            converged=False,
            diagnostics={
                "failure_reason": "all_lambda_candidates_failed",
                "lambda_candidate_traces": traces,
            },
        )

    best_result.method = "ppi_plus_plus"
    best_result.diagnostics.update(
        {
            "lambda_selection_mode": config.PPI_PLUS_PLUS_LAMBDA_MODE,
            "selected_covariance_trace": best_trace,
            "lambda_candidate_traces": traces,
        }
    )
    return best_result
