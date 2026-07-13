"""Integrity and numerical checks."""
from __future__ import annotations
import numpy as np
from config import LEARNER_IDS, PROBABILITY_EPS


def require_columns(columns, required, context="data"):
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f"Missing required columns in {context}: {missing}")


def check_galaxy_no_leakage(features):
    forbidden_exact = {"gz2_class"}
    target_prefix = "t01_smooth_or_features_a01_smooth_"
    forbidden_suffix = ("count", "weight", "fraction", "weighted_fraction", "debiased", "flag")
    bad = []
    for col in features:
        if col in forbidden_exact or (col.startswith(target_prefix) and col.endswith(forbidden_suffix)):
            bad.append(col)
        if col.startswith("t01_smooth_or_features_"):
            bad.append(col)
    if bad:
        raise ValueError(f"Galaxy Zoo target leakage detected: {sorted(set(bad))}")


def clip_probabilities(values):
    arr = np.asarray(values, dtype=float)
    return np.clip(arr, PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)


def check_interval(low, high):
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    if np.any(~np.isfinite(low)) or np.any(~np.isfinite(high)):
        raise ValueError("Non-finite confidence interval endpoint.")
    if np.any(low > high):
        raise ValueError("Confidence interval has lower endpoint above upper endpoint.")


def check_cross_predictions(pred_oof, pred_u_matrix, n, N, K):
    if np.asarray(pred_oof).shape != (n,):
        raise ValueError(f"OOF prediction shape mismatch: {np.asarray(pred_oof).shape}, expected {(n,)}")
    if np.asarray(pred_u_matrix).shape != (N, K):
        raise ValueError(f"Unlabelled cross-prediction shape mismatch: {np.asarray(pred_u_matrix).shape}, expected {(N, K)}")
    if np.any(~np.isfinite(pred_oof)) or np.any(~np.isfinite(pred_u_matrix)):
        raise ValueError("Cross-PPI predictions contain NaN or infinity.")


def check_learner_ids(ids):
    obsolete = {"L1", "L2", "L3", "L4"}
    if obsolete.intersection(ids):
        raise ValueError("Obsolete L1--L4 learner identifiers remain in the code path.")
    unknown = set(ids) - set(LEARNER_IDS)
    if unknown:
        raise ValueError(f"Unknown learner identifiers: {sorted(unknown)}")


def validate_loaded_data(experiment, y, X_target, X_learner, binary):
    """Fail fast on silent outcome-conversion and target-design failures."""
    arr = np.asarray(y)
    if arr.ndim != 1 or len(arr) < 2:
        raise ValueError(f"{experiment}: outcome must be a non-empty one-dimensional array.")
    if np.any(~np.isfinite(arr.astype(float))):
        raise ValueError(f"{experiment}: outcome contains NaN or infinity.")
    if binary:
        values = np.unique(arr.astype(int))
        if not np.array_equal(values, np.array([0, 1])):
            raise ValueError(
                f"{experiment}: binary outcome must contain both 0 and 1; found {values.tolist()}."
            )
    target = np.asarray(X_target, dtype=float)
    if target.shape[0] != len(arr) or np.any(~np.isfinite(target)):
        raise ValueError(f"{experiment}: invalid target-design matrix.")
    if len(X_learner) != len(arr):
        raise ValueError(f"{experiment}: learner features and outcome have different lengths.")
