"""Lightweight regression tests for toy's numerical integrity guards."""
from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from checks import run_preflight_checks
from main import _interval_covers


class TestCoreIntegrity(unittest.TestCase):
    def test_preflight_contract(self) -> None:
        run_preflight_checks()

    def test_exact_boundary_coverage_tolerates_roundoff(self) -> None:
        target = 0.9
        roundoff = 8.0 * np.finfo(float).eps
        self.assertTrue(_interval_covers(target + roundoff, target + roundoff, target))

    def test_materially_missed_interval_is_not_covered(self) -> None:
        self.assertFalse(_interval_covers(0.0, 0.1, 0.2))


if __name__ == "__main__":
    unittest.main()
