"""Deterministic information-flow and estimator-invariant audit.

Run with::

    python audit_checks.py

The script does not launch the Monte Carlo experiment.  It executes the A--H
checks requested for leakage, estimator boundaries, scenario fingerprints, and
small-sample hand calculations.  Results are written to
``audit/invariant_test_results.csv``.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math

import numpy as np
import pandas as pd

import config
from baselines import fit_classic
from data_generation import SimulationData, generate_replicate
from formulation import get_interval, get_scenario
from learner_proxy import PredictionBundle, generate_proxy
from main import MetricAccumulator, _record_result
from plotting import _summarise_for_plot
from ppi import fit_ppi, fit_weighted_ppi
from ppiplusplus import fit_ppi_plus_plus_v1, fit_ppi_plus_plus_v2

ROOT = Path(__file__).resolve().parent
AUDIT_DIR = ROOT / "audit"


def _vector(result, confidence_level: float = 0.95) -> np.ndarray:
    lower, upper = get_interval(result, confidence_level)
    return np.concatenate(
        [
            np.asarray(result.estimate).reshape(-1),
            np.asarray(result.covariance).reshape(-1),
            np.asarray(lower).reshape(-1),
            np.asarray(upper).reshape(-1),
        ]
    )


def _close(left, right, atol: float = 1e-9) -> bool:
    return bool(
        np.allclose(left, right, atol=atol, rtol=1e-8, equal_nan=False)
    )


def _synthetic_mean_data(
    y_labeled: np.ndarray,
    n_unlabeled: int,
) -> SimulationData:
    n_labeled = len(y_labeled)
    return SimulationData(
        seed=0,
        replicate_id=0,
        x_labeled=np.ones((n_labeled, 3), dtype=float),
        x_unlabeled=np.ones((n_unlabeled, 3), dtype=float),
        y_gaussian=np.asarray(y_labeled, dtype=float),
        y_logistic=np.resize(np.asarray([0.0, 1.0]), n_labeled),
        z_proxy_labeled=np.zeros(n_labeled),
        z_proxy_unlabeled=np.zeros(n_unlabeled),
    )


def run_audit() -> pd.DataFrame:
    rows: list[dict] = []

    # Test A: Classic independence from profile/predictions.
    for scenario_name in config.ACTIVE_SCENARIOS:
        scenario = get_scenario(scenario_name)
        data = generate_replicate(91001, 0)
        reference = _vector(fit_classic(scenario, data))
        passed = True
        for profile in config.ACTIVE_PROFILES:
            _ = generate_proxy(data, scenario, profile)
            passed = passed and _close(
                reference,
                _vector(fit_classic(scenario, data)),
                atol=0.0,
            )
        rows.append(
            {
                "test": "A",
                "case": scenario_name,
                "expected": "Classic estimate, covariance, and CI do not depend on profile",
                "actual": str(passed),
                "passed": passed,
            }
        )

    # Test B: high-quality prediction narrows a mean CI; independent noise worsens it.
    scenario = get_scenario("mean")
    y_labeled = np.asarray([-1.5, -0.5, 0.5, 1.5])
    n_unlabeled = 400
    y_unlabeled = np.tile(y_labeled, n_unlabeled // len(y_labeled))
    data = _synthetic_mean_data(y_labeled, n_unlabeled)
    perfect = PredictionBundle(
        profile="perfect",
        f_labeled=y_labeled.copy(),
        f_unlabeled=y_unlabeled.copy(),
        error_labeled=np.zeros_like(y_labeled),
        error_unlabeled=np.zeros_like(y_unlabeled),
    )
    rng = np.random.default_rng(91002)
    noisy = PredictionBundle(
        profile="noise",
        f_labeled=rng.normal(0.0, 50.0, len(y_labeled)),
        f_unlabeled=rng.normal(0.0, 50.0, n_unlabeled),
        error_labeled=np.zeros_like(y_labeled),
        error_unlabeled=np.zeros(n_unlabeled),
    )
    classic = fit_classic(scenario, data)
    good = fit_ppi(scenario, data, perfect)
    bad = fit_ppi(scenario, data, noisy)
    classic_var = float(classic.covariance[0, 0])
    good_var = float(good.covariance[0, 0])
    bad_var = float(bad.covariance[0, 0])
    passed = good_var < classic_var < bad_var
    rows.append(
        {
            "test": "B",
            "case": "mean",
            "expected": "Perfect prediction narrows; independent high noise worsens",
            "actual": (
                f"classic={classic_var:.8f}, perfect={good_var:.8f}, "
                f"noise={bad_var:.8f}"
            ),
            "passed": passed,
        }
    )

    # Test C: weighted-PPI endpoints.
    for scenario_name in config.ACTIVE_SCENARIOS:
        scenario = get_scenario(scenario_name)
        data = generate_replicate(91003, 0)
        prediction = generate_proxy(data, scenario, "P1")
        classic = fit_classic(scenario, data)
        lambda_zero = fit_weighted_ppi(
            scenario, data, prediction, lambda_=0.0, method="audit_lambda_zero"
        )
        ppi = fit_ppi(scenario, data, prediction)
        lambda_one = fit_weighted_ppi(
            scenario, data, prediction, lambda_=1.0, method="audit_lambda_one"
        )
        passed = _close(_vector(classic), _vector(lambda_zero)) and _close(
            _vector(ppi), _vector(lambda_one)
        )
        rows.append(
            {
                "test": "C",
                "case": scenario_name,
                "expected": "lambda=0 equals Classic; lambda=1 equals PPI",
                "actual": str(passed),
                "passed": passed,
            }
        )

    # Test D: inference must not depend on the evaluation truth vector.
    for scenario_name in config.ACTIVE_SCENARIOS:
        scenario = get_scenario(scenario_name)
        altered = replace(
            scenario,
            true_values=np.full_like(scenario.true_values, 12345.0),
        )
        data = generate_replicate(91004, 0)
        prediction = generate_proxy(data, scenario, "P1")
        prediction_altered = generate_proxy(data, altered, "P1")
        pairs = [
            (fit_classic(scenario, data), fit_classic(altered, data)),
            (
                fit_ppi(scenario, data, prediction),
                fit_ppi(altered, data, prediction_altered),
            ),
            (
                fit_ppi_plus_plus_v1(scenario, data, prediction),
                fit_ppi_plus_plus_v1(altered, data, prediction_altered),
            ),
            (
                fit_ppi_plus_plus_v2(scenario, data, prediction),
                fit_ppi_plus_plus_v2(altered, data, prediction_altered),
            ),
        ]
        passed = all(_close(_vector(left), _vector(right), 1e-7) for left, right in pairs)
        rows.append(
            {
                "test": "D",
                "case": scenario_name,
                "expected": "Changing truth alters evaluation only",
                "actual": str(passed),
                "passed": passed,
            }
        )

    # Test E: an injected unlabeled-label field is ignored by all estimators.
    for scenario_name in config.ACTIVE_SCENARIOS:
        scenario = get_scenario(scenario_name)
        data = generate_replicate(91005, 0)
        prediction = generate_proxy(data, scenario, "P1")
        before = [
            fit_classic(scenario, data),
            fit_ppi(scenario, data, prediction),
            fit_ppi_plus_plus_v1(scenario, data, prediction),
            fit_ppi_plus_plus_v2(scenario, data, prediction),
        ]
        data.y_unlabeled = np.random.default_rng(91006).normal(
            size=data.x_unlabeled.shape[0]
        )
        data.y_unlabeled = np.random.default_rng(91007).permutation(
            data.y_unlabeled
        )
        after = [
            fit_classic(scenario, data),
            fit_ppi(scenario, data, prediction),
            fit_ppi_plus_plus_v1(scenario, data, prediction),
            fit_ppi_plus_plus_v2(scenario, data, prediction),
        ]
        passed = all(
            _close(_vector(left), _vector(right), 1e-7)
            for left, right in zip(before, after)
        )
        rows.append(
            {
                "test": "E",
                "case": scenario_name,
                "expected": "Unlabeled labels are not part of the inference path",
                "actual": str(passed),
                "passed": passed,
            }
        )

    # Test F: scenario fingerprints.  Constant shifts are a known mean-PPI
    # invariance, so a nonconstant ramp is required to verify data flow.
    scenario = get_scenario("mean")
    data = generate_replicate(91008, 0)
    n = data.x_labeled.shape[0]
    N = data.x_unlabeled.shape[0]
    fingerprints = {
        "zero": PredictionBundle(
            "zero", np.zeros(n), np.zeros(N), np.zeros(n), np.zeros(N)
        ),
        "plus10": PredictionBundle(
            "plus10", np.full(n, 10.0), np.full(N, 10.0), np.zeros(n), np.zeros(N)
        ),
        "minus10": PredictionBundle(
            "minus10", np.full(n, -10.0), np.full(N, -10.0), np.zeros(n), np.zeros(N)
        ),
        "ramp": PredictionBundle(
            "ramp",
            np.arange(n, dtype=float),
            np.arange(N, dtype=float),
            np.zeros(n),
            np.zeros(N),
        ),
    }
    outputs = {
        name: fit_ppi(scenario, data, prediction)
        for name, prediction in fingerprints.items()
    }
    constant_invariance = _close(
        _vector(outputs["zero"]), _vector(outputs["plus10"])
    ) and _close(_vector(outputs["zero"]), _vector(outputs["minus10"]))
    ramp_changes = not _close(
        _vector(outputs["zero"]), _vector(outputs["ramp"])
    )
    passed = constant_invariance and ramp_changes
    rows.append(
        {
            "test": "F",
            "case": "mean",
            "expected": "Constant shifts cancel; a nonconstant fingerprint changes PPI",
            "actual": (
                f"constant_invariance={constant_invariance}, "
                f"ramp_changes={ramp_changes}"
            ),
            "passed": passed,
        }
    )

    # Test G: signature survives aggregate and plotting-summary transformations.
    scenario = get_scenario("mean")
    data = generate_replicate(91009, 0)
    prediction = generate_proxy(data, scenario, "P1")
    accumulator = MetricAccumulator()
    _record_result(
        accumulator,
        replicate_records=[],
        diagnostics=[],
        seed=0,
        replicate_id=0,
        scenario=scenario,
        profile="baseline",
        method="classic",
        result=fit_classic(scenario, data),
    )
    _record_result(
        accumulator,
        replicate_records=[],
        diagnostics=[],
        seed=0,
        replicate_id=0,
        scenario=scenario,
        profile="P1",
        method="ppi",
        result=fit_ppi(scenario, data, prediction),
    )
    aggregate = accumulator.to_frame()
    plot_frame = _summarise_for_plot(aggregate)
    passed = (
        "scenario_signature" in aggregate.columns
        and "scenario_signature" in plot_frame.columns
        and set(plot_frame["scenario_signature"])
        == {"mean:baseline", "mean:P1"}
    )
    rows.append(
        {
            "test": "G",
            "case": "result provenance",
            "expected": "Scenario signature reaches the plotting DataFrame",
            "actual": str(passed),
            "passed": passed,
        }
    )

    # Test H: n=4, N=6 hand calculation for all four inferential estimators.
    scenario = get_scenario("mean")
    y = np.asarray([1.0, 2.0, 4.0, 8.0])
    f_labeled = np.asarray([0.5, 2.5, 3.0, 7.0])
    f_unlabeled = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    data = _synthetic_mean_data(y, len(f_unlabeled))
    prediction = PredictionBundle(
        "manual",
        f_labeled,
        f_unlabeled,
        np.zeros_like(f_labeled),
        np.zeros_like(f_unlabeled),
    )

    classic_estimate = float(np.mean(y))
    classic_variance = float(np.var(y, ddof=1) / len(y))
    ppi_estimate = float(
        np.mean(y) + np.mean(f_unlabeled) - np.mean(f_labeled)
    )
    ppi_variance = float(
        np.var(y - f_labeled, ddof=1) / len(y)
        + np.var(f_unlabeled, ddof=1) / len(f_unlabeled)
    )
    classic = fit_classic(scenario, data)
    ppi = fit_ppi(scenario, data, prediction)
    base_ok = (
        math.isclose(classic.estimate[0], classic_estimate, abs_tol=1e-12)
        and math.isclose(
            classic.covariance[0, 0], classic_variance, abs_tol=1e-12
        )
        and math.isclose(ppi.estimate[0], ppi_estimate, abs_tol=1e-12)
        and math.isclose(ppi.covariance[0, 0], ppi_variance, abs_tol=1e-12)
    )

    best: tuple[float, float, float] | None = None
    for lambda_value in config.PPI_PLUS_PLUS_V2_LAMBDA_GRID:
        lambda_value = float(lambda_value)
        estimate = float(
            np.mean(y)
            + lambda_value
            * (np.mean(f_unlabeled) - np.mean(f_labeled))
        )
        variance = float(
            np.var(y - lambda_value * f_labeled, ddof=1) / len(y)
            + lambda_value**2
            * np.var(f_unlabeled, ddof=1)
            / len(f_unlabeled)
        )
        candidate = (variance, lambda_value, estimate)
        if best is None or candidate[0] < best[0] - 1e-15:
            best = candidate
    assert best is not None
    v2 = fit_ppi_plus_plus_v2(scenario, data, prediction)
    v2_ok = (
        math.isclose(v2.diagnostics["lambda_hat"], best[1], abs_tol=1e-12)
        and math.isclose(v2.estimate[0], best[2], abs_tol=1e-12)
        and math.isclose(v2.covariance[0, 0], best[0], abs_tol=1e-12)
    )

    all_predictions = np.concatenate([f_labeled, f_unlabeled])
    ppi_pilot = float(
        np.mean(f_unlabeled) + np.mean(y - f_labeled)
    )
    gradients = y - ppi_pilot
    predicted_gradients = f_labeled - ppi_pilot
    gradients_unlabeled = f_unlabeled - ppi_pilot
    centered = gradients - np.mean(gradients)
    centered_predicted = predicted_gradients - np.mean(predicted_gradients)
    covariance_gradient = float(
        (
            centered @ centered_predicted
            + centered_predicted @ centered
        )
        / len(y)
    )
    prediction_gradient_variance = float(np.cov(all_predictions))
    lambda_v1 = float(
        np.clip(
            covariance_gradient
            / (
                2.0
                * (1.0 + len(y) / len(f_unlabeled))
                * prediction_gradient_variance
            ),
            0.0,
            1.0,
        )
    )
    v1_estimate = float(
        lambda_v1 * np.mean(f_unlabeled)
        + np.mean(y - lambda_v1 * f_labeled)
    )
    v1_variance = float(
        np.var(lambda_v1 * f_unlabeled, ddof=0) / len(f_unlabeled)
        + np.var(y - lambda_v1 * f_labeled, ddof=0) / len(y)
    )
    v1 = fit_ppi_plus_plus_v1(scenario, data, prediction)
    lower, upper = get_interval(v1, 0.95)
    z_value = 1.959963984540054
    v1_ok = (
        math.isclose(v1.diagnostics["lambda_hat"], lambda_v1, abs_tol=1e-10)
        and math.isclose(v1.estimate[0], v1_estimate, abs_tol=1e-10)
        and math.isclose(
            (upper[0] - lower[0]) / (2.0 * z_value),
            math.sqrt(v1_variance),
            abs_tol=1e-10,
        )
    )
    passed = base_ok and v1_ok and v2_ok
    rows.append(
        {
            "test": "H",
            "case": "mean n=4 N=6",
            "expected": "Classic, PPI, PPI++V1, and PPI++V2 match hand calculation",
            "actual": (
                f"base={base_ok}, v1={v1_ok}, v2={v2_ok}, "
                f"lambda_v1={lambda_v1:.6f}, lambda_v2={best[1]:.3f}"
            ),
            "passed": passed,
        }
    )

    result = pd.DataFrame(rows)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(AUDIT_DIR / "invariant_test_results.csv", index=False)
    return result


if __name__ == "__main__":
    frame = run_audit()
    print(frame.to_string(index=False))
    passed = int(frame["passed"].sum())
    total = len(frame)
    print(f"\nPassed {passed}/{total} invariant checks.")
    if passed != total:
        raise SystemExit(1)
