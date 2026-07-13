# use python package ditectly
from __future__ import annotations
from importlib.metadata import version
import numpy as np
from scipy.stats import norm
from config import PPI_PY_REQUIRED_VERSION


def _api():
    try:
        import ppi_py
    except Exception as exc:
        raise RuntimeError("ppi-python is required for PPI++V1") from exc
    installed = version("ppi-python")
    if installed != PPI_PY_REQUIRED_VERSION:
        raise RuntimeError(
            f"ppi-python=={PPI_PY_REQUIRED_VERSION} required; found {installed}"
        )
    return ppi_py, installed


def _se_from_ci(estimate, ci, confidence=0.95):
    estimate = np.asarray(estimate, dtype=float)
    low = np.asarray(ci[0], dtype=float)
    high = np.asarray(ci[1], dtype=float)
    z = norm.ppf(0.5 + confidence / 2)
    return np.maximum((high - low) / (2 * z), 0)


def ppiplusv1_inference(task, X_inf, y_inf, pred_inf, X_u, pred_u):
    ppi_py, installed = _api()
    alpha = 0.05
    if task == "mean":
        estimate = np.atleast_1d(
            ppi_py.ppi_mean_pointestimate(y_inf, pred_inf, pred_u, lam=None)
        )
        ci = ppi_py.ppi_mean_ci(y_inf, pred_inf, pred_u, alpha=alpha, lam=None)
    elif task == "linear":
        estimate = np.asarray(
            ppi_py.ppi_ols_pointestimate(X_inf, y_inf, pred_inf, X_u, pred_u, lam=None),
            dtype=float,
        )
        ci = ppi_py.ppi_ols_ci(
            X_inf, y_inf, pred_inf, X_u, pred_u, alpha=alpha, lam=None
        )
    else:
        opts = {"ftol": 1e-10, "maxiter": 1000}
        estimate = np.asarray(
            ppi_py.ppi_logistic_pointestimate(
                X_inf, y_inf, pred_inf, X_u, pred_u, lam=None, optimizer_options=opts
            ),
            dtype=float,
        )
        ci = ppi_py.ppi_logistic_ci(
            X_inf,
            y_inf,
            pred_inf,
            X_u,
            pred_u,
            alpha=alpha,
            lam=None,
            optimizer_options=opts,
        )
    return {
        "estimate": estimate,
        "se": _se_from_ci(estimate, ci),
        "selected_lambda": np.nan,
        "lambda_source": "package_internal_not_exposed",
        "backend": "ppi-python",
        "package_version": installed,
    }
