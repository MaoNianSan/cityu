import numpy as np
from formulation import ppi_estimate_cov


def ppi_inference(task, X_inf, y_inf, pred_inf, X_unlabelled, pred_unlabelled):
    estimate, cov = ppi_estimate_cov(
        task, X_inf, y_inf, pred_inf, X_unlabelled, pred_unlabelled
    )
    return estimate, np.sqrt(np.maximum(np.diag(cov), 0)), cov
