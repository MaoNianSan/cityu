# some process include CI and others
from __future__ import annotations
import numpy as np
from scipy.special import expit
from scipy.stats import norm
from checks import check_interval


def _pinv(matrix, ridge=1e-10):
    matrix = np.asarray(matrix, dtype=float)
    return np.linalg.pinv(matrix + ridge * np.eye(matrix.shape[0]))


def _row_cov(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return np.array([[np.var(values, ddof=1)]])
    return np.atleast_2d(np.cov(values, rowvar=False, ddof=1))


def logistic_fit(X, y, max_iter=100, tol=1e-9):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta = np.zeros(X.shape[1], dtype=float)
    for _ in range(max_iter):
        mu = expit(np.clip(X @ beta, -30, 30))
        score = X.T @ (mu - y) / len(y)
        w = np.maximum(mu * (1 - mu), 1e-8)
        hess = (X.T * w) @ X / len(y)
        step = _pinv(hess) @ score
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def compute_full_truth(task, X, y):
    if task == "mean":
        return np.array([float(np.mean(y))])
    if task == "linear":
        return _pinv(X.T @ X) @ (X.T @ y)
    if task == "logistic":
        return logistic_fit(X, y)
    raise KeyError(task)


def ci_from_estimate_se(estimate, se, confidence_level):
    estimate = np.asarray(estimate, dtype=float)
    se = np.asarray(se, dtype=float)
    z = norm.ppf(0.5 + confidence_level / 2.0)
    low, high = estimate - z * se, estimate + z * se
    check_interval(low, high)
    return low, high


def classic_estimate_se(task, X, y):
    y = np.asarray(y, dtype=float)
    if task == "mean":
        return np.array([y.mean()]), np.array([y.std(ddof=1) / np.sqrt(len(y))])
    if task == "linear":
        beta = _pinv(X.T @ X) @ (X.T @ y)
        residual = y - X @ beta
        H = X.T @ X / len(y)
        scores = X * residual[:, None]
        cov = _pinv(H) @ (_row_cov(scores) / len(y)) @ _pinv(H)
        return beta, np.sqrt(np.maximum(np.diag(cov), 0))
    beta = logistic_fit(X, y)
    mu = expit(np.clip(X @ beta, -30, 30))
    W = np.maximum(mu * (1 - mu), 1e-8)
    H = (X.T * W) @ X / len(y)
    scores = X * (y - mu)[:, None]
    cov = _pinv(H) @ (_row_cov(scores) / len(y)) @ _pinv(H)
    return beta, np.sqrt(np.maximum(np.diag(cov), 0))


def naive_estimate_se(task, X_u, pred_u):
    return classic_estimate_se(task, X_u, np.asarray(pred_u, dtype=float))


def ppi_estimate_cov(task, X_i, y_i, pred_i, X_u, pred_u):
    n, N = len(y_i), len(pred_u)
    if task == "mean":
        estimate = np.array([np.mean(pred_u) + np.mean(y_i - pred_i)])
        var = np.var(pred_u, ddof=1) / N + np.var(y_i - pred_i, ddof=1) / n
        return estimate, np.array([[var]])
    if task == "linear":
        H = X_u.T @ X_u / N
        rhs = X_u.T @ pred_u / N + X_i.T @ (y_i - pred_i) / n
        beta = _pinv(H) @ rhs
        score_u = X_u * ((X_u @ beta - pred_u)[:, None])
        score_i = X_i * ((pred_i - y_i)[:, None])
    else:
        beta = np.zeros(X_u.shape[1])
        for _ in range(100):
            mu_u = expit(np.clip(X_u @ beta, -30, 30))
            score = X_u.T @ (mu_u - pred_u) / N + X_i.T @ (pred_i - y_i) / n
            w_u = np.maximum(mu_u * (1 - mu_u), 1e-8)
            H = (X_u.T * w_u) @ X_u / N
            step = _pinv(H) @ score
            new = beta - step
            if np.max(np.abs(new - beta)) < 1e-9:
                beta = new
                break
            beta = new
        mu_u = expit(np.clip(X_u @ beta, -30, 30))
        w_u = np.maximum(mu_u * (1 - mu_u), 1e-8)
        H = (X_u.T * w_u) @ X_u / N
        score_u = X_u * ((mu_u - pred_u)[:, None])
        score_i = X_i * ((pred_i - y_i)[:, None])
    meat = _row_cov(score_u) / N + _row_cov(score_i) / n
    cov = _pinv(H) @ meat @ _pinv(H)
    return beta, cov


def ppiv2_estimate_cov(task, X_i, y_i, pred_i, X_u, pred_u, lam):
    n, N = len(y_i), len(pred_u)
    if task == "mean":
        estimate = np.array([np.mean(y_i) + lam * (np.mean(pred_u) - np.mean(pred_i))])
        var = (
            np.var(y_i - lam * pred_i, ddof=1) / n + lam**2 * np.var(pred_u, ddof=1) / N
        )
        return estimate, np.array([[var]])
    if task == "linear":
        H_i = X_i.T @ X_i / n
        H_u = X_u.T @ X_u / N
        H = (1 - lam) * H_i + lam * H_u
        rhs = X_i.T @ y_i / n + lam * (X_u.T @ pred_u / N - X_i.T @ pred_i / n)
        beta = _pinv(H) @ rhs
        score_i = X_i * (((1 - lam) * (X_i @ beta) - y_i + lam * pred_i)[:, None])
        score_u = lam * X_u * (((X_u @ beta) - pred_u)[:, None])
    else:
        beta = logistic_fit(X_i, y_i)
        for _ in range(100):
            mu_i = expit(np.clip(X_i @ beta, -30, 30))
            mu_u = expit(np.clip(X_u @ beta, -30, 30))
            score_i_scalar = (1 - lam) * mu_i - y_i + lam * pred_i
            score_u_scalar = lam * (mu_u - pred_u)
            score = X_i.T @ score_i_scalar / n + X_u.T @ score_u_scalar / N
            wi = np.maximum(mu_i * (1 - mu_i), 1e-8)
            wu = np.maximum(mu_u * (1 - mu_u), 1e-8)
            H = (1 - lam) * (X_i.T * wi) @ X_i / n + lam * (X_u.T * wu) @ X_u / N
            step = _pinv(H) @ score
            new = beta - step
            if np.max(np.abs(new - beta)) < 1e-9:
                beta = new
                break
            beta = new
        mu_i = expit(np.clip(X_i @ beta, -30, 30))
        mu_u = expit(np.clip(X_u @ beta, -30, 30))
        wi = np.maximum(mu_i * (1 - mu_i), 1e-8)
        wu = np.maximum(mu_u * (1 - mu_u), 1e-8)
        H = (1 - lam) * (X_i.T * wi) @ X_i / n + lam * (X_u.T * wu) @ X_u / N
        score_i = X_i * (((1 - lam) * mu_i - y_i + lam * pred_i)[:, None])
        score_u = lam * X_u * ((mu_u - pred_u)[:, None])
    meat = _row_cov(score_i) / n + _row_cov(score_u) / N
    cov = _pinv(H) @ meat @ _pinv(H)
    return beta, cov
