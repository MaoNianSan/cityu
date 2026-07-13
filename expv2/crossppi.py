from __future__ import annotations
from importlib.metadata import version
import numpy as np
from scipy.stats import norm
from config import PPI_PY_REQUIRED_VERSION


def _api():
    try:
        import ppi_py
    except Exception as exc:
        raise RuntimeError("ppi-python is required for Cross-PPI") from exc
    installed = version("ppi-python")
    if installed != PPI_PY_REQUIRED_VERSION:
        raise RuntimeError(
            f"ppi-python=={PPI_PY_REQUIRED_VERSION} required; found {installed}"
        )
    return ppi_py, installed


def _se_from_ci(ci, confidence=0.95):
    low = np.asarray(ci[0], dtype=float)
    high = np.asarray(ci[1], dtype=float)
    return np.maximum((high - low) / (2 * norm.ppf(0.5 + confidence / 2)), 0)


def crossppi_inference(task, X_l, y_l, pred_oof, X_u, pred_u_matrix):
    ppi_py, installed = _api()
    alpha = 0.05
    if task == "mean":
        estimate = np.atleast_1d(
            ppi_py.crossppi_mean_pointestimate(y_l, pred_oof, pred_u_matrix)
        )
        ci = ppi_py.crossppi_mean_ci(y_l, pred_oof, pred_u_matrix, alpha=alpha)
    elif task == "linear":
        estimate = np.asarray(
            ppi_py.crossppi_ols_pointestimate(X_l, y_l, pred_oof, X_u, pred_u_matrix),
            dtype=float,
        )
        ci = ppi_py.crossppi_ols_ci(X_l, y_l, pred_oof, X_u, pred_u_matrix, alpha=alpha)
    else:
        opts = {"ftol": 1e-10, "maxiter": 1000}
        estimate = np.asarray(
            ppi_py.crossppi_logistic_pointestimate(
                X_l, y_l, pred_oof, X_u, pred_u_matrix, optimizer_options=opts
            ),
            dtype=float,
        )
        ci = ppi_py.crossppi_logistic_ci(
            X_l, y_l, pred_oof, X_u, pred_u_matrix, alpha=alpha, optimizer_options=opts
        )
    return {
        "estimate": estimate,
        "se": _se_from_ci(ci),
        "backend": "ppi-python",
        "package_version": installed,
    }
