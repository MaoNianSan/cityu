"""Reserved Cross-PPI interface.

Cross-PPI is intentionally not part of the current preliminary experiment.
The later implementation should fit a fold-specific learner f^{(-k)} rather
than reuse the controlled oracle proxy outputs in learner_proxy.py.
"""
from __future__ import annotations

from typing import Any


def fit_cross_ppi(*args: Any, **kwargs: Any) -> None:
    """Placeholder for the later K-fold Cross-PPI implementation."""
    raise NotImplementedError(
        "Cross-PPI is reserved in this package but disabled for the current "
        "preliminary experiment. Set up fold-specific learner training before enabling it."
    )
