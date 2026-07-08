"""Generation and optional caching of one paired simulation replicate."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config
from formulation import ScenarioSpec, get_scenario


Array = np.ndarray


@dataclass
class SimulationData:
    """Shared data used by every method and proxy profile within one replicate."""

    seed: int
    replicate_id: int
    x_labeled: Array
    x_unlabeled: Array
    y_gaussian: Array
    y_logistic: Array
    u_labeled: Array
    u_unlabeled: Array

    def outcome_for(self, scenario: ScenarioSpec | str) -> Array:
        scenario_name = scenario if isinstance(scenario, str) else scenario.name
        if scenario_name in {"mean", "lr"}:
            return self.y_gaussian
        if scenario_name == "logistic_glm":
            return self.y_logistic
        raise ValueError(f"Unknown scenario {scenario_name!r}.")


def _rng(seed: int, replicate_id: int, stream_id: int) -> np.random.Generator:
    """Create an order-invariant random stream for one component of a replicate."""
    sequence = np.random.SeedSequence([int(seed), int(replicate_id), int(stream_id)])
    return np.random.default_rng(sequence)


def _draw_design(n_rows: int, rng: np.random.Generator) -> Array:
    x1 = rng.uniform(config.X1_LOW, config.X1_HIGH, size=n_rows)
    x2 = rng.binomial(1, config.X2_PROBABILITY, size=n_rows).astype(float)
    return np.column_stack([np.ones(n_rows, dtype=float), x1, x2])


def generate_replicate(seed: int, replicate_id: int) -> SimulationData:
    """Generate paired labelled/unlabelled data and controlled proxy randomness."""
    x_labeled = _draw_design(config.N_LABELED, _rng(seed, replicate_id, 11))
    x_unlabeled = _draw_design(config.N_UNLABELED, _rng(seed, replicate_id, 12))

    gaussian_spec = get_scenario("lr")
    logistic_spec = get_scenario("logistic_glm")
    y_gaussian = gaussian_spec.generate_outcome(x_labeled, _rng(seed, replicate_id, 21))
    y_logistic = logistic_spec.generate_outcome(x_labeled, _rng(seed, replicate_id, 22))

    # P1 and P4 share these U draws; only their amplitudes differ.
    u_labeled = _rng(seed, replicate_id, 31).uniform(-1.0, 1.0, size=config.N_LABELED)
    u_unlabeled = _rng(seed, replicate_id, 32).uniform(-1.0, 1.0, size=config.N_UNLABELED)

    return SimulationData(
        seed=int(seed),
        replicate_id=int(replicate_id),
        x_labeled=x_labeled,
        x_unlabeled=x_unlabeled,
        y_gaussian=y_gaussian,
        y_logistic=y_logistic,
        u_labeled=u_labeled,
        u_unlabeled=u_unlabeled,
    )


def save_replicate(data: SimulationData, destination: Path) -> None:
    """Persist one generated input replicate as a compact NPZ archive."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        seed=np.asarray([data.seed], dtype=int),
        replicate_id=np.asarray([data.replicate_id], dtype=int),
        x_labeled=data.x_labeled,
        x_unlabeled=data.x_unlabeled,
        y_gaussian=data.y_gaussian,
        y_logistic=data.y_logistic,
        u_labeled=data.u_labeled,
        u_unlabeled=data.u_unlabeled,
    )


def load_replicate(source: Path) -> SimulationData:
    """Load a replicate saved by :func:`save_replicate`."""
    with np.load(source) as archive:
        return SimulationData(
            seed=int(archive["seed"][0]),
            replicate_id=int(archive["replicate_id"][0]),
            x_labeled=archive["x_labeled"],
            x_unlabeled=archive["x_unlabeled"],
            y_gaussian=archive["y_gaussian"],
            y_logistic=archive["y_logistic"],
            u_labeled=archive["u_labeled"],
            u_unlabeled=archive["u_unlabeled"],
        )
