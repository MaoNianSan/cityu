"""Controlled proxy-output regimes P1--P4 for the preliminary experiment."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from data_generation import SimulationData
from formulation import ScenarioSpec


Array = np.ndarray


@dataclass
class PredictionBundle:
    """Pseudo learner predictions and their known controlled errors."""

    profile: str
    f_labeled: Array
    f_unlabeled: Array
    error_labeled: Array
    error_unlabeled: Array


def _error(profile: str, x: Array, u: Array) -> Array:
    if profile == "P1":
        return config.P1_NOISE_AMPLITUDE * u
    if profile == "P2":
        return np.full(x.shape[0], config.P2_CONSTANT_SHIFT, dtype=float)
    if profile == "P3":
        # x[:, 2] is X2 because x=(1, X1, X2)^T.
        return config.P3_SIGNED_SHIFT * (2.0 * x[:, 2] - 1.0)
    if profile == "P4":
        return config.P4_NOISE_AMPLITUDE * u
    raise ValueError(f"Unknown proxy profile {profile!r}.")


def generate_proxy(
    data: SimulationData,
    scenario: ScenarioSpec,
    profile: str,
) -> PredictionBundle:
    """Construct f=z+e under one controlled learner-quality profile."""
    if profile not in config.ACTIVE_PROFILES:
        raise ValueError(f"Profile {profile!r} is not active in config.ACTIVE_PROFILES.")

    z_labeled = scenario.conditional_mean(data.x_labeled)
    z_unlabeled = scenario.conditional_mean(data.x_unlabeled)
    e_labeled = _error(profile, data.x_labeled, data.u_labeled)
    e_unlabeled = _error(profile, data.x_unlabeled, data.u_unlabeled)
    f_labeled = z_labeled + e_labeled
    f_unlabeled = z_unlabeled + e_unlabeled

    if scenario.family == "logistic":
        lower = min(float(f_labeled.min()), float(f_unlabeled.min()))
        upper = max(float(f_labeled.max()), float(f_unlabeled.max()))
        if lower < 0.0 or upper > 1.0:
            raise ValueError(
                "A logistic pseudo outcome fell outside [0, 1]. "
                "The implementation does not clip proxy predictions."
            )

    return PredictionBundle(
        profile=profile,
        f_labeled=f_labeled,
        f_unlabeled=f_unlabeled,
        error_labeled=e_labeled,
        error_unlabeled=e_unlabeled,
    )
