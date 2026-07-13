from __future__ import annotations
import numpy as np


def compute_quality(binary: bool, y_true, prediction):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prediction, dtype=float)
    value = float(np.mean((y - p) ** 2))
    return ("Brier" if binary else "MSE"), value
