"""Strict ppi-python==0.2.3 wrapper for PPI++V1.

The package's ``lam=None`` point-estimate and CI entry points do not share a
single resolved lambda for OLS and logistic regression: the CI path recomputes
lambda after obtaining the point estimate.  This wrapper resolves lambda once
from the ordinary PPI (lambda=1) pilot and passes the same scalar explicitly to
both public package calls.
"""
from __future__ import annotations
from importlib.metadata import version
import numpy as np
from scipy.stats import norm
from config import PPI_PY_REQUIRED_VERSION


def _api():
    try:
        import ppi_py
        from ppi_py.ppi import _calc_lam_glm, _ols_get_stats, _logistic_get_stats
    except Exception as exc:
        raise RuntimeError("ppi-python is required for PPI++V1") from exc
    installed = version("ppi-python")
    if installed != PPI_PY_REQUIRED_VERSION:
        raise RuntimeError(
            f"ppi-python=={PPI_PY_REQUIRED_VERSION} required; found {installed}"
        )
    return ppi_py, _calc_lam_glm, _ols_get_stats, _logistic_get_stats, installed


def _se_from_ci(ci, confidence=0.95):
    low = np.asarray(ci[0], dtype=float)
    high = np.asarray(ci[1], dtype=float)
    z = norm.ppf(0.5 + confidence / 2)
    return np.maximum((high - low) / (2 * z), 0)


def _select_lambda_once(task, X_inf, y_inf, pred_inf, X_u, pred_u, opts=None):
    """Reproduce the package's first lambda update and expose the scalar."""
    ppi_py, calc_lam, ols_stats, logistic_stats, installed = _api()
    y_inf = np.asarray(y_inf, dtype=float)
    pred_inf = np.asarray(pred_inf, dtype=float)
    pred_u = np.asarray(pred_u, dtype=float)

    if task == "mean":
        pilot = np.atleast_1d(
            ppi_py.ppi_mean_pointestimate(y_inf, pred_inf, pred_u, lam=1.0)
        )
        Y = y_inf.reshape(-1, 1)
        Yhat = pred_inf.reshape(-1, 1)
        Yhat_u = pred_u.reshape(-1, 1)
        grads = Y - pilot.reshape(1, -1)
        grads_hat = Yhat - pilot.reshape(1, -1)
        grads_hat_u = Yhat_u - pilot.reshape(1, -1)
        lam = calc_lam(
            grads,
            grads_hat,
            grads_hat_u,
            np.eye(pilot.size),
            coord=None,
            clip=True,
            optim_mode="overall",
        )
    elif task == "linear":
        X_inf = np.asarray(X_inf, dtype=float)
        X_u = np.asarray(X_u, dtype=float)
        pilot = np.asarray(
            ppi_py.ppi_ols_pointestimate(
                X_inf, y_inf, pred_inf, X_u, pred_u, lam=1.0
            ),
            dtype=float,
        )
        stats = ols_stats(
            pilot,
            X_inf,
            y_inf,
            pred_inf,
            X_u,
            pred_u,
            use_unlabeled=True,
        )
        lam = calc_lam(*stats, coord=None, clip=True)
    elif task == "logistic":
        X_inf = np.asarray(X_inf, dtype=float)
        X_u = np.asarray(X_u, dtype=float)
        pilot = np.asarray(
            ppi_py.ppi_logistic_pointestimate(
                X_inf,
                y_inf,
                pred_inf,
                X_u,
                pred_u,
                lam=1.0,
                optimizer_options=opts,
            ),
            dtype=float,
        )
        stats = logistic_stats(
            pilot,
            X_inf,
            y_inf,
            pred_inf,
            X_u,
            pred_u,
            use_unlabeled=True,
        )
        lam = calc_lam(*stats, coord=None, clip=True)
    else:
        raise KeyError(task)

    lam = float(np.asarray(lam).squeeze())
    if not np.isfinite(lam) or not (0.0 <= lam <= 1.0):
        raise RuntimeError(f"Invalid PPI++V1 lambda selected: {lam}")
    return lam, installed


def ppiplusv1_inference(task, X_inf, y_inf, pred_inf, X_u, pred_u):
    ppi_py, _, _, _, _ = _api()
    alpha = 0.05
    opts = {"ftol": 1e-10, "maxiter": 1000} if task == "logistic" else None
    lam, installed = _select_lambda_once(
        task, X_inf, y_inf, pred_inf, X_u, pred_u, opts=opts
    )

    if task == "mean":
        estimate = np.atleast_1d(
            ppi_py.ppi_mean_pointestimate(y_inf, pred_inf, pred_u, lam=lam)
        )
        ci = ppi_py.ppi_mean_ci(
            y_inf, pred_inf, pred_u, alpha=alpha, lam=lam
        )
    elif task == "linear":
        estimate = np.asarray(
            ppi_py.ppi_ols_pointestimate(
                X_inf, y_inf, pred_inf, X_u, pred_u, lam=lam
            ),
            dtype=float,
        )
        ci = ppi_py.ppi_ols_ci(
            X_inf, y_inf, pred_inf, X_u, pred_u, alpha=alpha, lam=lam
        )
    else:
        estimate = np.asarray(
            ppi_py.ppi_logistic_pointestimate(
                X_inf,
                y_inf,
                pred_inf,
                X_u,
                pred_u,
                lam=lam,
                optimizer_options=opts,
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
            lam=lam,
            optimizer_options=opts,
        )

    midpoint = (np.asarray(ci[0], dtype=float) + np.asarray(ci[1], dtype=float)) / 2
    if not np.allclose(estimate, midpoint, rtol=1e-9, atol=1e-10):
        raise RuntimeError(
            "PPI++V1 estimate is not the midpoint of its fixed-lambda package CI."
        )
    return {
        "estimate": estimate,
        "se": _se_from_ci(ci),
        "selected_lambda": lam,
        "lambda_source": "package_formula_resolved_once",
        "backend": "ppi-python",
        "package_version": installed,
    }
