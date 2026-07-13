from __future__ import annotations
import numpy as np
from config import PPI_PLUS_PLUS_LAMBDA_GRID
from formulation import ppiv2_estimate_cov


def ppiplusv2_inference(task, X_inf, y_inf, pred_inf, X_unlabelled, pred_unlabelled):
    diagnostics = []
    best = None
    for lam in PPI_PLUS_PLUS_LAMBDA_GRID:
        estimate, cov = ppiv2_estimate_cov(
            task, X_inf, y_inf, pred_inf, X_unlabelled, pred_unlabelled, float(lam)
        )
        trace = float(np.trace(cov))
        diagnostics.append({"lambda": float(lam), "cov_trace": trace})
        if np.isfinite(trace) and (best is None or trace < best[0] - 1e-15):
            best = (trace, float(lam), estimate, cov)
    if best is None:
        raise RuntimeError("PPI++V2 failed for every lambda.")
    _, lam, estimate, cov = best
    # Exact implementation boundary audit.
    est0, cov0 = ppiv2_estimate_cov(
        task, X_inf, y_inf, pred_inf, X_unlabelled, pred_unlabelled, 0.0
    )
    est1, cov1 = ppiv2_estimate_cov(
        task, X_inf, y_inf, pred_inf, X_unlabelled, pred_unlabelled, 1.0
    )
    return {
        "estimate": estimate,
        "se": np.sqrt(np.maximum(np.diag(cov), 0)),
        "cov": cov,
        "selected_lambda": lam,
        "grid_diagnostics": diagnostics,
        "lambda0_estimate": est0,
        "lambda1_estimate": est1,
        "lambda0_cov": cov0,
        "lambda1_cov": cov1,
    }
