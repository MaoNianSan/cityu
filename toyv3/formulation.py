from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.special import expit

import config

Array = np.ndarray


@dataclass(frozen=True)
class ScenarioSpec:
    """Scenario-specific target and response-generation contract."""

    name: str
    family: str
    target_names: tuple[str, ...]
    true_values: Array
    conditional_mean_fn: Callable[[Array], Array]
    outcome_fn: Callable[[Array, np.random.Generator], Array]

    def conditional_mean(self, x: Array) -> Array:
        return self.conditional_mean_fn(x)

    def generate_outcome(self, x: Array, rng: np.random.Generator) -> Array:
        return self.outcome_fn(x, rng)

    def linear_predictor(self, x: Array) -> Array:
        return design_signal(x)


@dataclass
class EstimatorResult:
    """Unified return object for every estimator."""

    method: str
    estimate: Array
    covariance: Array
    converged: bool = True
    diagnostics: dict = field(default_factory=dict)
    intervals: dict[float, tuple[Array, Array]] | None = None


def interval_key(confidence_level: float) -> float:
    return round(float(confidence_level), 6)


def get_interval(result: EstimatorResult, confidence_level: float) -> tuple[Array, Array]:
    key = interval_key(confidence_level)
    if result.intervals is not None and key in result.intervals:
        lower, upper = result.intervals[key]
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    z_value = float(__import__("scipy.stats", fromlist=["norm"]).norm.ppf(0.5 + key / 2.0))
    se = np.sqrt(np.maximum(np.diag(np.asarray(result.covariance, dtype=float)), 0.0))
    return result.estimate - z_value * se, result.estimate + z_value * se


def design_signal(x: Array) -> Array:
    """Return eta(X) = X beta* for the shared design matrix X=(1, X1, X2)."""
    beta = np.asarray(config.TRUE_BETA, dtype=float)
    return np.asarray(x, dtype=float) @ beta


def _gaussian_outcome(x: Array, rng: np.random.Generator) -> Array:
    mean = design_signal(x)
    return mean + rng.normal(0.0, config.GAUSSIAN_NOISE_SD, size=x.shape[0])


def _logistic_mean(x: Array) -> Array:
    return expit(design_signal(x))


def _logistic_outcome(x: Array, rng: np.random.Generator) -> Array:
    probability = _logistic_mean(x)
    return rng.binomial(1, probability, size=x.shape[0]).astype(float)


def _mean_true_value() -> float:
    # E[X1] = 0 and E[X2] = X2_PROBABILITY.
    beta0, _, beta2 = config.TRUE_BETA
    return float(beta0 + beta2 * config.X2_PROBABILITY)


def get_scenario(name: str) -> ScenarioSpec:
    """Return the immutable mathematical contract for one active scenario."""
    beta = np.asarray(config.TRUE_BETA, dtype=float)
    if name == "mean":
        return ScenarioSpec(
            name="mean",
            family="mean",
            target_names=("theta",),
            true_values=np.asarray([_mean_true_value()], dtype=float),
            conditional_mean_fn=design_signal,
            outcome_fn=_gaussian_outcome,
        )
    if name == "lr":
        return ScenarioSpec(
            name="lr",
            family="linear",
            target_names=("beta_0", "beta_1", "beta_2"),
            true_values=beta.copy(),
            conditional_mean_fn=design_signal,
            outcome_fn=_gaussian_outcome,
        )
    if name == "logistic_glm":
        return ScenarioSpec(
            name="logistic_glm",
            family="logistic",
            target_names=("beta_0", "beta_1", "beta_2"),
            true_values=beta.copy(),
            conditional_mean_fn=_logistic_mean,
            outcome_fn=_logistic_outcome,
        )
    raise ValueError(f"Unknown scenario {name!r}.")


def all_scenarios() -> tuple[ScenarioSpec, ...]:
    return tuple(get_scenario(name) for name in config.ACTIVE_SCENARIOS)


def display_name(name: str) -> str:
    mapping = {
        "mean": "Mean estimation",
        "lr": "Linear regression",
        "logistic_glm": "Logistic GLM",
    }
    return mapping.get(name, name)
