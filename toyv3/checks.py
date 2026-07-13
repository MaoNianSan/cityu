"""Preflight mathematical and numerical checks for the toy package."""

from __future__ import annotations

import numpy as np
from scipy.special import logit

import config
from baselines import fit_classic
from data_generation import generate_replicate
from formulation import all_scenarios, get_scenario
from learner_proxy import generate_proxy
from ppi import fit_ppi, fit_weighted_ppi
from ppiplusplus import fit_ppi_plus_plus_v1


def _assert_close(
    left: np.ndarray, right: np.ndarray, label: str, atol: float = 1e-9
) -> None:
    if not np.allclose(left, right, atol=atol, rtol=1e-8, equal_nan=False):
        raise AssertionError(f"Check failed: {label}.")


def run_preflight_checks() -> None:
    """Run deterministic checks before any fast/full simulation starts."""
    data = generate_replicate(seed=98765, replicate_id=0)
    data_again = generate_replicate(seed=98765, replicate_id=0)

    _assert_close(
        data.x_labeled, data_again.x_labeled, "same seed reproduces X_labeled"
    )
    _assert_close(
        data.x_unlabeled, data_again.x_unlabeled, "same seed reproduces X_unlabeled"
    )
    _assert_close(
        data.y_gaussian, data_again.y_gaussian, "same seed reproduces Gaussian Y"
    )
    _assert_close(
        data.y_logistic, data_again.y_logistic, "same seed reproduces logistic Y"
    )
    _assert_close(data.z_proxy_labeled, data_again.z_proxy_labeled, "same seed reproduces labelled proxy Z", atol=0.0)
    _assert_close(data.z_proxy_unlabeled, data_again.z_proxy_unlabeled, "same seed reproduces unlabelled proxy Z", atol=0.0)
    if data.z_proxy_labeled.shape != (config.N_LABELED,) or data.z_proxy_unlabeled.shape != (config.N_UNLABELED,):
        raise AssertionError("Proxy Z shape is incorrect.")
    if not (np.isfinite(data.z_proxy_labeled).all() and np.isfinite(data.z_proxy_unlabeled).all()):
        raise AssertionError("Proxy Z contains non-finite values.")
    different = generate_replicate(seed=98765, replicate_id=1)
    if np.array_equal(data.z_proxy_labeled, different.z_proxy_labeled):
        raise AssertionError("Different replicates reused labelled proxy Z.")

    for scenario in all_scenarios():
        p1 = generate_proxy(data, scenario, "P1")
        p2 = generate_proxy(data, scenario, "P2")
        p3 = generate_proxy(data, scenario, "P3")
        p4 = generate_proxy(data, scenario, "P4")

        _assert_close(
            p2.error_labeled**2,
            np.full(config.N_LABELED, config.P2**2),
            "P2 labelled squared error",
        )
        _assert_close(
            p3.error_labeled**2,
            np.full(config.N_LABELED, config.P3**2),
            "P3 labelled squared error",
        )
        _assert_close(
            p2.error_labeled**2,
            p3.error_labeled**2,
            "P2/P3 labelled equal pointwise squared error",
        )
        _assert_close(
            p2.error_unlabeled**2,
            np.full(config.N_UNLABELED, config.P2**2),
            "P2 unlabelled squared error",
        )
        _assert_close(
            p3.error_unlabeled**2,
            np.full(config.N_UNLABELED, config.P3**2),
            "P3 unlabelled squared error",
        )
        _assert_close(
            p2.error_unlabeled**2,
            p3.error_unlabeled**2,
            "P2/P3 unlabelled equal pointwise squared error",
        )
        if scenario.family == "logistic":
            for proxy in (p1, p2, p3, p4):
                if not (
                    np.all(proxy.f_labeled > 0.0) and np.all(proxy.f_labeled < 1.0)
                ):
                    raise AssertionError("Logistic labelled proxy is outside [0, 1].")
                if not (
                    np.all(proxy.f_unlabeled > 0.0)
                    and np.all(proxy.f_unlabeled < 1.0)
                ):
                    raise AssertionError("Logistic unlabelled proxy is outside [0, 1].")
            eta_l = scenario.linear_predictor(data.x_labeled)
            eta_u = scenario.linear_predictor(data.x_unlabeled)
            _assert_close(p1.error_labeled, p1.f_labeled - scenario.conditional_mean(data.x_labeled), "logistic P1 error definition")
            _assert_close(p4.error_labeled, p4.f_labeled - scenario.conditional_mean(data.x_labeled), "logistic P4 error definition")
            ratio = float(config.P4 / config.P1)
            _assert_close(logit(p4.f_labeled) - eta_l, ratio * (logit(p1.f_labeled) - eta_l), "logistic labelled latent ratio", atol=1e-10)
            _assert_close(logit(p4.f_unlabeled) - eta_u, ratio * (logit(p1.f_unlabeled) - eta_u), "logistic unlabelled latent ratio", atol=1e-10)
        else:
            _assert_close(p1.error_labeled, config.P1 * data.z_proxy_labeled, "Gaussian P1 labelled normal error", atol=1e-12)
            _assert_close(p4.error_labeled, config.P4 * data.z_proxy_labeled, "Gaussian P4 labelled normal error", atol=1e-12)
            ratio = float(config.P4 / config.P1)
            _assert_close(p4.error_labeled, ratio * p1.error_labeled, "Gaussian labelled scale ratio", atol=1e-12)
            _assert_close(p4.error_unlabeled, ratio * p1.error_unlabeled, "Gaussian unlabelled scale ratio", atol=1e-12)

        # Endpoint identities for the PPI++ family.
        classic = fit_classic(scenario, data)
        lambda_zero = fit_weighted_ppi(
            scenario, data, p1, lambda_=0.0, method="endpoint_lambda_zero"
        )
        _assert_close(
            classic.estimate,
            lambda_zero.estimate,
            f"lambda=0 equals classic estimate for {scenario.name}",
        )
        _assert_close(
            classic.covariance,
            lambda_zero.covariance,
            f"lambda=0 equals classic covariance for {scenario.name}",
        )

        ppi = fit_ppi(scenario, data, p1)
        lambda_one = fit_weighted_ppi(
            scenario, data, p1, lambda_=1.0, method="endpoint_lambda_one"
        )
        _assert_close(
            ppi.estimate,
            lambda_one.estimate,
            f"lambda=1 equals PPI estimate for {scenario.name}",
        )
        _assert_close(
            ppi.covariance,
            lambda_one.covariance,
            f"lambda=1 equals PPI covariance for {scenario.name}",
        )

        # PPI++V1 must use one coherent lambda for its point estimate and all
        # direct package intervals.
        v1 = fit_ppi_plus_plus_v1(scenario, data, p1)
        if not v1.converged:
            raise AssertionError(
                f"PPI++V1 preflight failed for {scenario.name}: "
                f"{v1.diagnostics.get('exception')}"
            )
        if v1.diagnostics.get("lambda_hat") is None:
            raise AssertionError("PPI++V1 did not expose its selected lambda.")
        for confidence_level in config.CONFIDENCE_LEVELS:
            lower, upper = v1.intervals[round(float(confidence_level), 6)]
            _assert_close(
                0.5 * (lower + upper),
                v1.estimate,
                f"PPI++V1 estimate equals CI midpoint for {scenario.name}/{confidence_level}",
                atol=1e-8,
            )

    # Explicitly check the mean target under the stated DGP.
    mean_spec = get_scenario("mean")
    _assert_close(
        mean_spec.true_values, np.asarray([-0.05]), "mean target theta* = -0.05"
    )


if __name__ == "__main__":
    run_preflight_checks()
    print("All preflight checks passed.")
