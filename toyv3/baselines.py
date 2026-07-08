from __future__ import annotations

from data_generation import SimulationData
from formulation import EstimatorResult, ScenarioSpec
from learner_proxy import PredictionBundle
from ppi import fit_single_sample_model


def fit_classic(scenario: ScenarioSpec, data: SimulationData) -> EstimatorResult:
    """Fit inference using labelled outcomes only."""
    return fit_single_sample_model(
        scenario=scenario,
        x=data.x_labeled,
        y=data.outcome_for(scenario),
        method="classic",
    )  ## Naive statistic inference


def fit_naive_ml(
    scenario: ScenarioSpec,
    data: SimulationData,
    prediction: PredictionBundle,
) -> EstimatorResult:
    """Treat pseudo outcomes on unlabelled data as if they were gold labels."""
    return fit_single_sample_model(
        scenario=scenario,
        x=data.x_unlabeled,
        y=prediction.f_unlabeled,
        method="naive_ml",
    )  ## Only ML learner will be used for inference
