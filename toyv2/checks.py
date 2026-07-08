"""Preflight mathematical and numerical checks for the toy package."""

from __future__ import annotations

import numpy as np

import config
from baselines import fit_classic
from data_generation import generate_replicate
from formulation import all_scenarios, get_scenario
from learner_proxy import generate_proxy
from ppi import fit_ppi, fit_weighted_ppi


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
        _assert_close(
            p4.error_labeled,
            10.0 * p1.error_labeled,
            "P4/P1 labelled shared U and amplitude ratio",
        )
        _assert_close(
            p4.error_unlabeled,
            10.0 * p1.error_unlabeled,
            "P4/P1 unlabelled shared U and amplitude ratio",
        )

        if scenario.family == "logistic":
            for proxy in (p1, p2, p3, p4):
                if not (
                    np.all(proxy.f_labeled >= 0.0) and np.all(proxy.f_labeled <= 1.0)
                ):
                    raise AssertionError("Logistic labelled proxy is outside [0, 1].")
                if not (
                    np.all(proxy.f_unlabeled >= 0.0)
                    and np.all(proxy.f_unlabeled <= 1.0)
                ):
                    raise AssertionError("Logistic unlabelled proxy is outside [0, 1].")

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

    # Explicitly check the mean target under the stated DGP.
    mean_spec = get_scenario("mean")
    _assert_close(
        mean_spec.true_values, np.asarray([-0.05]), "mean target theta* = -0.05"
    )


if __name__ == "__main__":
    run_preflight_checks()
    print("All preflight checks passed.")
