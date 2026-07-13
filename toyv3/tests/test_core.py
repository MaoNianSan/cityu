"""Lightweight regression tests for toy's numerical integrity guards."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checks import run_preflight_checks
from main import _interval_covers
from formulation import EstimatorResult, get_interval
from formulation import get_scenario
from data_generation import generate_replicate
from learner_proxy import generate_proxy
from ppiplusplus import fit_ppi_plus_plus_v1


class TestCoreIntegrity(unittest.TestCase):
    def test_preflight_contract(self) -> None:
        run_preflight_checks()

    def test_exact_boundary_coverage_tolerates_roundoff(self) -> None:
        target = 0.9
        roundoff = 8.0 * np.finfo(float).eps
        self.assertTrue(_interval_covers(target + roundoff, target + roundoff, target))

    def test_materially_missed_interval_is_not_covered(self) -> None:
        self.assertFalse(_interval_covers(0.0, 0.1, 0.2))

    def test_direct_interval_takes_priority(self) -> None:
        result = EstimatorResult("x", np.array([0.0]), np.array([[100.0]]), intervals={0.95: (np.array([-1.0]), np.array([2.0]))})
        lower, upper = get_interval(result, 0.95000000001)
        np.testing.assert_array_equal(lower, [-1.0])
        np.testing.assert_array_equal(upper, [2.0])

    def test_ppi_python_called_at_every_confidence_level(self) -> None:
        scenario = get_scenario("mean")
        data = generate_replicate(123, 0)
        proxy = generate_proxy(data, scenario, "P1")
        alphas = []
        def ci(*args, **kwargs):
            alphas.append(kwargs["alpha"])
            return np.array([-kwargs["alpha"]]), np.array([kwargs["alpha"]])
        with patch("ppiplusplus._ppi_python_functions", return_value=(lambda *a, **k: np.array([0.0]), ci)):
            result = fit_ppi_plus_plus_v1(scenario, data, proxy)
        np.testing.assert_allclose(alphas, [0.10, 0.05, 0.025])
        self.assertEqual(set(result.intervals), {0.90, 0.95, 0.975})
        self.assertEqual(
            result.diagnostics["interval_source"],
            "ppi_python_fixed_selected_lambda",
        )
        self.assertIsNotNone(result.diagnostics["lambda_hat"])


if __name__ == "__main__":
    unittest.main()
