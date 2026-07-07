"""Standard PPI and the shared weighted PPI-family solver.

The private/general solver is intentionally shared with PPI++ so that
lambda=1 is exactly standard PPI and lambda=0 is exactly classic inference.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.special import expit

import config
from data_generation import SimulationData
from formulation import EstimatorResult, ScenarioSpec
from learner_proxy import PredictionBundle


Array = np.ndarray


def _row_covariance(rows: Array) -> Array:
    rows = np.asarray(rows, dtype=float)
    if rows.ndim == 1:
        rows = rows[:, None]
    p = rows.shape[1]
    if rows.shape[0] <= 1:
        return np.zeros((p, p), dtype=float)
    centered = rows - rows.mean(axis=0, keepdims=True)
    return (centered.T @ centered) / (rows.shape[0] - 1)


def _sandwich_covariance(
    hessian: Array,
    psi_labeled: Array | None,
    n_labeled: int | None,
    psi_unlabeled: Array | None,
    n_unlabeled: int | None,
) -> tuple[Array, float]:
    """Compute H^{-1} Omega H^{-T} for independent labelled/unlabelled means."""
    hessian = np.asarray(hessian, dtype=float)
    p = hessian.shape[0]
    omega = np.zeros((p, p), dtype=float)
    if psi_labeled is not None and n_labeled is not None:
        omega += _row_covariance(psi_labeled) / n_labeled
    if psi_unlabeled is not None and n_unlabeled is not None:
        omega += _row_covariance(psi_unlabeled) / n_unlabeled

    condition_number = float(np.linalg.cond(hessian))
    h_inv = np.linalg.solve(hessian, np.eye(p))
    covariance = h_inv @ omega @ h_inv.T
    covariance = 0.5 * (covariance + covariance.T)
    return covariance, condition_number


def _failed_result(method: str, p: int, reason: str, diagnostics: dict | None = None) -> EstimatorResult:
    diagnostics = dict(diagnostics or {})
    diagnostics["failure_reason"] = reason
    return EstimatorResult(
        method=method,
        estimate=np.full(p, np.nan),
        covariance=np.full((p, p), np.nan),
        converged=False,
        diagnostics=diagnostics,
    )


def _newton_solver(
    callback: Callable[[Array], tuple[Array, Array, float]],
    p: int,
    initial_beta: Array | None = None,
) -> tuple[Array, bool, dict]:
    """Damped Newton solver for a convex logistic estimating objective."""
    beta = np.zeros(p, dtype=float) if initial_beta is None else np.asarray(initial_beta, dtype=float).copy()
    max_condition_number = 0.0

    for iteration in range(1, config.GLM_MAX_ITER + 1):
        gradient, hessian, objective = callback(beta)
        if not (np.all(np.isfinite(gradient)) and np.all(np.isfinite(hessian)) and np.isfinite(objective)):
            return beta, False, {
                "iterations": iteration,
                "condition_number": max_condition_number,
                "failure_reason": "non_finite_gradient_hessian_or_objective",
            }

        gradient_norm = float(np.max(np.abs(gradient)))
        condition_number = float(np.linalg.cond(hessian))
        max_condition_number = max(max_condition_number, condition_number)
        if not np.isfinite(condition_number) or condition_number > config.MAX_HESSIAN_CONDITION_NUMBER:
            return beta, False, {
                "iterations": iteration,
                "condition_number": condition_number,
                "failure_reason": "ill_conditioned_hessian",
            }
        if gradient_norm <= config.GLM_TOL:
            return beta, True, {
                "iterations": iteration,
                "condition_number": condition_number,
                "gradient_max_abs": gradient_norm,
            }

        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return beta, False, {
                "iterations": iteration,
                "condition_number": condition_number,
                "failure_reason": "singular_hessian",
            }

        step_scale = 1.0
        accepted = False
        for _ in range(config.GLM_LINESEARCH_MAX_STEPS):
            candidate = beta - step_scale * step
            _, _, candidate_objective = callback(candidate)
            if np.isfinite(candidate_objective) and candidate_objective <= objective + 1e-12:
                beta = candidate
                accepted = True
                break
            step_scale *= 0.5

        if not accepted:
            return beta, False, {
                "iterations": iteration,
                "condition_number": condition_number,
                "failure_reason": "line_search_failed",
            }

    gradient, hessian, _ = callback(beta)
    return beta, False, {
        "iterations": config.GLM_MAX_ITER,
        "condition_number": float(np.linalg.cond(hessian)),
        "gradient_max_abs": float(np.max(np.abs(gradient))),
        "failure_reason": "max_iterations_reached",
    }


def fit_single_sample_model(
    scenario: ScenarioSpec,
    x: Array,
    y: Array,
    method: str,
) -> EstimatorResult:
    """Fit classic or naive-ML inference on one sample with one outcome vector."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.shape[0]

    if scenario.family == "mean":
        estimate = np.asarray([float(np.mean(y))])
        variance = float(np.var(y, ddof=1) / n)
        return EstimatorResult(
            method=method,
            estimate=estimate,
            covariance=np.asarray([[variance]]),
            diagnostics={"lambda_hat": 0.0 if method == "classic" else None},
        )

    if scenario.family == "linear":
        hessian = (x.T @ x) / n
        try:
            estimate = np.linalg.solve(hessian, (x.T @ y) / n)
            residual = x @ estimate - y
            psi = x * residual[:, None]
            covariance, condition_number = _sandwich_covariance(hessian, psi, n, None, None)
        except np.linalg.LinAlgError:
            return _failed_result(method, x.shape[1], "singular_linear_hessian")
        return EstimatorResult(
            method=method,
            estimate=estimate,
            covariance=covariance,
            diagnostics={"condition_number": condition_number, "lambda_hat": 0.0 if method == "classic" else None},
        )

    if scenario.family == "logistic":
        p = x.shape[1]

        def callback(beta: Array) -> tuple[Array, Array, float]:
            eta = x @ beta
            mu = expit(eta)
            gradient = (x.T @ (mu - y)) / n
            weight = mu * (1.0 - mu)
            hessian = (x.T @ (x * weight[:, None])) / n
            objective = float(np.mean(np.logaddexp(0.0, eta) - y * eta))
            return gradient, hessian, objective

        estimate, converged, diagnostics = _newton_solver(callback, p)
        if not converged:
            return _failed_result(method, p, diagnostics.get("failure_reason", "logistic_solver_failure"), diagnostics)
        gradient, hessian, _ = callback(estimate)
        psi = x * (expit(x @ estimate) - y)[:, None]
        try:
            covariance, condition_number = _sandwich_covariance(hessian, psi, n, None, None)
        except np.linalg.LinAlgError:
            return _failed_result(method, p, "singular_logistic_hessian", diagnostics)
        diagnostics.update(
            {
                "condition_number": condition_number,
                "gradient_max_abs": float(np.max(np.abs(gradient))),
                "lambda_hat": 0.0 if method == "classic" else None,
            }
        )
        return EstimatorResult(method=method, estimate=estimate, covariance=covariance, diagnostics=diagnostics)

    raise ValueError(f"Unsupported family {scenario.family!r}.")


def fit_weighted_ppi(
    scenario: ScenarioSpec,
    data: SimulationData,
    prediction: PredictionBundle | None,
    lambda_: float,
    method: str,
    initial_beta: Array | None = None,
) -> EstimatorResult:
    """Fit the PPI family at a fixed lambda in [0, 1].

    lambda=0 gives classic inference; lambda=1 gives standard PPI.
    """
    if not 0.0 <= float(lambda_) <= 1.0:
        raise ValueError("lambda_ must lie in [0, 1].")
    lambda_ = float(lambda_)
    y_labeled = data.outcome_for(scenario)
    x_labeled = data.x_labeled
    x_unlabeled = data.x_unlabeled
    n = x_labeled.shape[0]
    N = x_unlabeled.shape[0]

    if lambda_ == 0.0:
        # The proxy is mathematically irrelevant at lambda=0.
        f_labeled = np.zeros(n, dtype=float)
        f_unlabeled = np.zeros(N, dtype=float)
    else:
        if prediction is None:
            raise ValueError("prediction is required whenever lambda_ is nonzero.")
        f_labeled = np.asarray(prediction.f_labeled, dtype=float)
        f_unlabeled = np.asarray(prediction.f_unlabeled, dtype=float)

    if scenario.family == "mean":
        estimate_value = float(np.mean(y_labeled) + lambda_ * (np.mean(f_unlabeled) - np.mean(f_labeled)))
        residual_labeled = y_labeled - lambda_ * f_labeled
        variance = float(np.var(residual_labeled, ddof=1) / n + (lambda_ ** 2) * np.var(f_unlabeled, ddof=1) / N)
        return EstimatorResult(
            method=method,
            estimate=np.asarray([estimate_value]),
            covariance=np.asarray([[variance]]),
            diagnostics={"lambda_hat": lambda_},
        )

    if scenario.family == "linear":
        hessian = (1.0 - lambda_) * (x_labeled.T @ x_labeled) / n + lambda_ * (x_unlabeled.T @ x_unlabeled) / N
        rhs = (x_labeled.T @ (y_labeled - lambda_ * f_labeled)) / n + lambda_ * (x_unlabeled.T @ f_unlabeled) / N
        try:
            estimate = np.linalg.solve(hessian, rhs)
            psi_labeled = x_labeled * (
                (1.0 - lambda_) * (x_labeled @ estimate) - y_labeled + lambda_ * f_labeled
            )[:, None]
            psi_unlabeled = lambda_ * x_unlabeled * ((x_unlabeled @ estimate) - f_unlabeled)[:, None]
            covariance, condition_number = _sandwich_covariance(
                hessian, psi_labeled, n, psi_unlabeled, N
            )
        except np.linalg.LinAlgError:
            return _failed_result(method, x_labeled.shape[1], "singular_weighted_linear_hessian", {"lambda_hat": lambda_})
        return EstimatorResult(
            method=method,
            estimate=estimate,
            covariance=covariance,
            diagnostics={"lambda_hat": lambda_, "condition_number": condition_number},
        )

    if scenario.family == "logistic":
        p = x_labeled.shape[1]

        def callback(beta: Array) -> tuple[Array, Array, float]:
            eta_labeled = x_labeled @ beta
            eta_unlabeled = x_unlabeled @ beta
            mu_labeled = expit(eta_labeled)
            mu_unlabeled = expit(eta_unlabeled)

            labelled_scalar = (1.0 - lambda_) * mu_labeled - y_labeled + lambda_ * f_labeled
            unlabeled_scalar = lambda_ * (mu_unlabeled - f_unlabeled)
            gradient = (x_labeled.T @ labelled_scalar) / n + (x_unlabeled.T @ unlabeled_scalar) / N

            w_labeled = mu_labeled * (1.0 - mu_labeled)
            w_unlabeled = mu_unlabeled * (1.0 - mu_unlabeled)
            hessian = (
                (1.0 - lambda_) * (x_labeled.T @ (x_labeled * w_labeled[:, None])) / n
                + lambda_ * (x_unlabeled.T @ (x_unlabeled * w_unlabeled[:, None])) / N
            )

            labelled_objective = (
                (1.0 - lambda_) * np.mean(np.logaddexp(0.0, eta_labeled))
                - np.mean((y_labeled - lambda_ * f_labeled) * eta_labeled)
            )
            unlabeled_objective = lambda_ * np.mean(
                np.logaddexp(0.0, eta_unlabeled) - f_unlabeled * eta_unlabeled
            )
            return gradient, hessian, float(labelled_objective + unlabeled_objective)

        estimate, converged, diagnostics = _newton_solver(callback, p, initial_beta=initial_beta)
        diagnostics["lambda_hat"] = lambda_
        if not converged:
            return _failed_result(method, p, diagnostics.get("failure_reason", "weighted_logistic_solver_failure"), diagnostics)

        gradient, hessian, _ = callback(estimate)
        mu_labeled = expit(x_labeled @ estimate)
        mu_unlabeled = expit(x_unlabeled @ estimate)
        psi_labeled = x_labeled * (
            (1.0 - lambda_) * mu_labeled - y_labeled + lambda_ * f_labeled
        )[:, None]
        psi_unlabeled = lambda_ * x_unlabeled * (mu_unlabeled - f_unlabeled)[:, None]
        try:
            covariance, condition_number = _sandwich_covariance(
                hessian, psi_labeled, n, psi_unlabeled, N
            )
        except np.linalg.LinAlgError:
            return _failed_result(method, p, "singular_weighted_logistic_hessian", diagnostics)
        diagnostics.update(
            {
                "condition_number": condition_number,
                "gradient_max_abs": float(np.max(np.abs(gradient))),
            }
        )
        return EstimatorResult(method=method, estimate=estimate, covariance=covariance, diagnostics=diagnostics)

    raise ValueError(f"Unsupported family {scenario.family!r}.")


def fit_ppi(
    scenario: ScenarioSpec,
    data: SimulationData,
    prediction: PredictionBundle,
) -> EstimatorResult:
    """Fit standard PPI, equivalently the weighted PPI family at lambda=1."""
    return fit_weighted_ppi(
        scenario=scenario,
        data=data,
        prediction=prediction,
        lambda_=1.0,
        method="ppi",
    )
