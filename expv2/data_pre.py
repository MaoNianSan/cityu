"""Dataset loading, cleaning, feature construction, and empirical truth inputs."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from config import DATASET_CONFIG
from checks import require_columns, check_galaxy_no_leakage, validate_loaded_data


def _add_intercept(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, list[str]]:
    values = frame[columns].astype(float).to_numpy()
    return np.column_stack([np.ones(len(frame)), values]), ["intercept", *columns]


def load_experiment(experiment: str) -> dict:
    if experiment not in DATASET_CONFIG:
        raise KeyError(f"Unknown experiment: {experiment}")
    cfg = DATASET_CONFIG[experiment]
    path = Path(cfg["path"])
    if not path.exists():
        raise FileNotFoundError(path)

    if experiment == "a_mean":
        usecols = [cfg["outcome"], *cfg["learner_features"]]
        df = pd.read_csv(path, usecols=usecols)
        require_columns(df.columns, usecols, experiment)
        check_galaxy_no_leakage(cfg["learner_features"])
        df = df.dropna(subset=[cfg["outcome"]]).reset_index(drop=True)
        y = df[cfg["outcome"]].astype(int).to_numpy()
        X_target = np.ones((len(df), 1), dtype=float)
        parameter_names = ["smooth_proportion"]

    elif experiment == "b_lr":
        raw_cols = ["PINCP", "AGEP", "SEX", "SCHL", "MAR", "COW", "WKHP", "POBP", "RAC1P", "HICOV", "ESR"]
        df = pd.read_csv(path, usecols=raw_cols)
        require_columns(df.columns, raw_cols, experiment)
        df = df.dropna(subset=["PINCP", "AGEP", "SEX"]).copy()
        df["SEX_MALE"] = (df["SEX"].astype(float) == 1.0).astype(int)
        df = df.reset_index(drop=True)
        y = df["PINCP"].astype(float).to_numpy()
        X_target, parameter_names = _add_intercept(df, cfg["target_features"])

    else:  # c_glm
        usecols = [cfg["outcome"], *cfg["learner_features"]]
        df = pd.read_csv(path, usecols=usecols)
        require_columns(df.columns, usecols, experiment)
        df = df.dropna(subset=[cfg["outcome"], *cfg["target_features"]]).reset_index(drop=True)
        y = (pd.to_numeric(df[cfg["outcome"]], errors="coerce") > 50000).astype(int).to_numpy()
        X_target, parameter_names = _add_intercept(df, cfg["target_features"])

    X_learner = df[cfg["learner_features"]].copy()
    validate_loaded_data(experiment, y, X_target, X_learner, bool(cfg["binary"]))
    return {
        "experiment": experiment,
        "task": cfg["task"],
        "target_name": cfg["target_name"],
        "y": y,
        "X_target": X_target,
        "X_learner": X_learner,
        "parameter_names": parameter_names,
        "display_parameters": cfg.get("display_parameters", parameter_names),
        "numeric_features": list(cfg["numeric_features"]),
        "categorical_features": list(cfg["categorical_features"]),
        "learner_features": list(cfg["learner_features"]),
        "quality_metric": cfg["quality_metric"],
        "binary": bool(cfg["binary"]),
        "source_path": str(path),
    }
