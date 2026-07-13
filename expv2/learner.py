"""Common learner registry, standard preprocessing, within-class tuning, and cross-fitting."""
from __future__ import annotations
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from config import LEARNER_GRIDS, LEARNER_LABELS, TUNING_VALIDATION_RATIO, PROBABILITY_EPS
from learner_quality import compute_quality
from checks import clip_probabilities, check_cross_predictions


@dataclass
class TunedLearner:
    model: object
    learner: str
    learner_label: str
    learner_model: str
    tuned_params: dict
    validation_score: float


def _onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _preprocessor(learner: str, numeric: list[str], categorical: list[str]):
    if learner == "lin":
        num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", _onehot())])
        return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)], remainder="drop")
    num = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)], remainder="drop", sparse_threshold=0.0)


def _estimator(learner: str, binary: bool, seed: int):
    if learner == "lin":
        if binary:
            return LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
        return Ridge(solver="lsqr")
    if learner == "dt":
        cls = DecisionTreeClassifier if binary else DecisionTreeRegressor
        return cls(random_state=seed)
    if learner == "rf":
        cls = RandomForestClassifier if binary else RandomForestRegressor
        return cls(random_state=seed, n_jobs=1, max_features=0.7, max_samples=0.8)
    if learner == "gb":
        cls = HistGradientBoostingClassifier if binary else HistGradientBoostingRegressor
        return cls(random_state=seed, early_stopping=False)
    raise KeyError(learner)


def model_name(learner: str, binary: bool):
    names = {
        ("lin", True): "Logistic Regression", ("lin", False): "Ridge Regression",
        ("dt", True): "Decision Tree Classifier", ("dt", False): "Decision Tree Regressor",
        ("rf", True): "Random Forest Classifier", ("rf", False): "Random Forest Regressor",
        ("gb", True): "Histogram Gradient Boosting Classifier", ("gb", False): "Histogram Gradient Boosting Regressor",
    }
    return names[(learner, binary)]


def _grid(learner: str, binary: bool):
    if learner == "lin":
        return LEARNER_GRIDS["lin_binary" if binary else "lin_continuous"]
    return LEARNER_GRIDS[learner]


def _predict(model, X, binary: bool):
    if binary:
        return clip_probabilities(model.predict_proba(X)[:, 1])
    return np.asarray(model.predict(X), dtype=float)


def tune_and_fit(
    learner: str,
    X: pd.DataFrame,
    y,
    train_idx,
    numeric_features,
    categorical_features,
    binary: bool,
    seed: int,
) -> TunedLearner:
    train_idx = np.asarray(train_idx)
    stratify = np.asarray(y)[train_idx] if binary else None
    fit_idx, val_idx = train_test_split(
        train_idx,
        test_size=TUNING_VALIDATION_RATIO,
        random_state=seed,
        stratify=stratify,
        shuffle=True,
    )
    base = Pipeline([
        ("preprocess", _preprocessor(learner, numeric_features, categorical_features)),
        ("model", _estimator(learner, binary, seed)),
    ])
    best_score = np.inf
    best_params = None
    for params in _grid(learner, binary):
        candidate = clone(base).set_params(**params)
        candidate.fit(X.iloc[fit_idx], np.asarray(y)[fit_idx])
        pred = _predict(candidate, X.iloc[val_idx], binary)
        _, score = compute_quality(binary, np.asarray(y)[val_idx], pred)
        if score < best_score - 1e-15:
            best_score = score
            best_params = dict(params)
    final_model = clone(base).set_params(**best_params)
    final_model.fit(X.iloc[train_idx], np.asarray(y)[train_idx])
    return TunedLearner(
        model=final_model,
        learner=learner,
        learner_label=LEARNER_LABELS[learner],
        learner_model=model_name(learner, binary),
        tuned_params=best_params,
        validation_score=float(best_score),
    )


def predict(tuned: TunedLearner, X: pd.DataFrame, idx, binary: bool):
    return _predict(tuned.model, X.iloc[np.asarray(idx)], binary)


def build_cross_predictions(
    learner: str,
    X: pd.DataFrame,
    y,
    labelled_idx,
    unlabelled_idx,
    folds,
    numeric_features,
    categorical_features,
    binary: bool,
    seed: int,
):
    labelled_idx = np.asarray(labelled_idx)
    unlabelled_idx = np.asarray(unlabelled_idx)
    position = {int(idx): pos for pos, idx in enumerate(labelled_idx)}
    pred_oof = np.full(len(labelled_idx), np.nan, dtype=float)
    pred_u_matrix = np.empty((len(unlabelled_idx), len(folds)), dtype=float)
    fold_details = []
    for fold_id, (train_idx, hold_idx) in enumerate(folds):
        # Hyperparameter selection and refitting use only fold-excluded labels.
        tuned = tune_and_fit(
            learner, X, y, train_idx, numeric_features, categorical_features,
            binary, seed + 1009 * (fold_id + 1),
        )
        hold_pred = predict(tuned, X, hold_idx, binary)
        for idx, value in zip(hold_idx, hold_pred):
            pred_oof[position[int(idx)]] = value
        pred_u_matrix[:, fold_id] = predict(tuned, X, unlabelled_idx, binary)
        fold_details.append({
            "fold_id": fold_id,
            "n_train": int(len(train_idx)),
            "n_holdout": int(len(hold_idx)),
            "tuned_params": tuned.tuned_params,
            "validation_score": tuned.validation_score,
        })
    check_cross_predictions(pred_oof, pred_u_matrix, len(labelled_idx), len(unlabelled_idx), len(folds))
    return {
        "pred_oof": pred_oof,
        "pred_u_matrix": pred_u_matrix,
        "pred_u_bar": pred_u_matrix.mean(axis=1),
        "fold_details": fold_details,
    }
