"""CLI and experiment orchestration for real-data EXP1."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")

from concurrent.futures import ThreadPoolExecutor, as_completed
from threadpoolctl import threadpool_limits
from importlib.metadata import PackageNotFoundError, version

# Prevent nested BLAS/OpenMP oversubscription before NumPy/scikit-learn imports.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd

from config import *
from checks import check_learner_ids
from data_pre import load_experiment
from split import make_label_unlabel_split, make_ppi_train_inf_split, make_crossppi_folds
from learner import tune_and_fit, predict, build_cross_predictions, model_name
from learner_quality import compute_quality
from formulation import compute_full_truth, ci_from_estimate_se, classic_estimate_se
from baselines import classic_inference, naive_ml_inference
from ppi import ppi_inference
from ppiplusv1 import ppiplusv1_inference
from ppiplusv2 import ppiplusv2_inference
from crossppi import crossppi_inference
from metrics import WARNING_COLUMNS, aggregate_results, summarize_ppiv2_tuning_effect
from plot import plot_learner_quality, plot_inference_performance_95, plot_coverage_calibration_by_parameter

# Apply one process-wide native thread limit. This is not nested inside worker
# threads and therefore avoids the thread-safety issues of per-worker contexts.
_NATIVE_THREAD_LIMITER = threadpool_limits(limits=1)


def _dirs(mode, experiment):
    base = OUTPUT_DIR / mode / experiment
    return base, base / "figure", base / "table", base / "other"


def _prepare_dirs(mode, experiment):
    base, fig, table, other = _dirs(mode, experiment)
    if CLEAN_OUTPUT_BEFORE_RUN and base.exists():
        shutil.rmtree(base)
    for p in (fig, table, other):
        p.mkdir(parents=True, exist_ok=True)
    return base, fig, table, other


def _package_version():
    try:
        return version("ppi-python")
    except PackageNotFoundError:
        return "not-installed"


def _rows_for(
    method,
    method_label,
    estimate,
    se,
    truth,
    parameter_names,
    experiment,
    replicate,
    target,
    learner_info,
    quality,
    selected_lambda=np.nan,
    lambda_source="",
    backend="internal",
    package_version="",
    status="ok",
    warning="",
    sizes=None,
):
    rows = []
    estimate = np.asarray(estimate, dtype=float)
    se = np.asarray(se, dtype=float)
    qmetric, qvalue = quality if quality else ("", np.nan)
    sizes = sizes or {}
    for cl in CONFIDENCE_LEVELS:
        low, high = ci_from_estimate_se(estimate, se, cl)
        for j, param in enumerate(parameter_names):
            rows.append(
                {
                    "experiment": experiment,
                    "replicate": replicate,
                    "target": target,
                    "parameter": param,
                    "learner": learner_info.get("learner", "reference"),
                    "learner_label": learner_info.get("learner_label", "Reference"),
                    "learner_model": learner_info.get("learner_model", "None"),
                    "method": method,
                    "method_label": method_label,
                    "confidence_level": cl,
                    "estimate": float(estimate[j]),
                    "truth": float(truth[j]),
                    "ci_low": float(low[j]),
                    "ci_high": float(high[j]),
                    "ci_width": float(high[j] - low[j]),
                    "covered": bool(low[j] <= truth[j] <= high[j]),
                    "quality_metric": qmetric,
                    "quality_value": qvalue,
                    "selected_lambda": selected_lambda,
                    "lambda_source": lambda_source,
                    "backend": backend,
                    "package_version": package_version,
                    "status": status,
                    "warning": warning,
                    "tuned_params": json.dumps(learner_info.get("tuned_params", {}), sort_keys=True),
                    "validation_score": learner_info.get("validation_score", np.nan),
                    **sizes,
                }
            )
    return rows


def _failed_rows(method, experiment, replicate, data, truth, learner_info, quality, warning, sizes):
    rows = []
    qmetric, qvalue = quality
    for cl in CONFIDENCE_LEVELS:
        for param, t in zip(data["parameter_names"], truth):
            rows.append(
                {
                    "experiment": experiment,
                    "replicate": replicate,
                    "target": data["target_name"],
                    "parameter": param,
                    "learner": learner_info["learner"],
                    "learner_label": learner_info["learner_label"],
                    "learner_model": learner_info["learner_model"],
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "confidence_level": cl,
                    "estimate": np.nan,
                    "truth": float(t),
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "ci_width": np.nan,
                    "covered": False,
                    "quality_metric": qmetric,
                    "quality_value": qvalue,
                    "selected_lambda": np.nan,
                    "lambda_source": "",
                    "backend": "ppi-python" if method in {"ppi_plus_plus_v1", "cross_ppi"} else "internal",
                    "package_version": _package_version(),
                    "status": "failed",
                    "warning": warning,
                    "tuned_params": json.dumps(learner_info.get("tuned_params", {}), sort_keys=True),
                    "validation_score": learner_info.get("validation_score", np.nan),
                    **sizes,
                }
            )
    return rows


def _prepare_learner_artifacts(
    learner_id,
    data,
    replicate,
    seed,
    labelled_idx,
    unlabelled_idx,
    train_idx,
    inf_idx,
    folds,
):
    """Fit split and fold-specific learners and generate all predictions.

    This is the only part run in parallel. Inference is performed serially after
    all artifacts are available, avoiding severe memory/GIL contention from
    simultaneous PPI++V2 grids and ppi-python Cross-PPI confidence routines.
    """
    y = data["y"]
    X = data["X_learner"]
    binary = data["binary"]

    tuned = tune_and_fit(
        learner_id,
        X,
        y,
        train_idx,
        data["numeric_features"],
        data["categorical_features"],
        binary,
        seed + 37,
    )
    info = {
        "learner": learner_id,
        "learner_label": tuned.learner_label,
        "learner_model": tuned.learner_model,
        "tuned_params": tuned.tuned_params,
        "validation_score": tuned.validation_score,
    }
    pred_inf = predict(tuned, X, inf_idx, binary)
    pred_u = predict(tuned, X, unlabelled_idx, binary)
    quality = compute_quality(binary, y[inf_idx], pred_inf)

    cross = build_cross_predictions(
        learner_id,
        X,
        y,
        labelled_idx,
        unlabelled_idx,
        folds,
        data["numeric_features"],
        data["categorical_features"],
        binary,
        seed + 7919,
    )
    cross_quality = compute_quality(binary, y[labelled_idx], cross["pred_oof"])[1]

    return {
        "info": info,
        "pred_inf": pred_inf,
        "pred_u": pred_u,
        "quality": quality,
        "cross": cross,
        "cross_quality": cross_quality,
    }


def _infer_from_artifacts(
    artifacts,
    data,
    replicate,
    labelled_idx,
    unlabelled_idx,
    train_idx,
    inf_idx,
    truth,
    sizes,
):
    y = data["y"]
    Xt = data["X_target"]
    task = data["task"]
    binary = data["binary"]
    info = artifacts["info"]
    pred_inf = artifacts["pred_inf"]
    pred_u = artifacts["pred_u"]
    quality = artifacts["quality"]
    cross = artifacts["cross"]

    Xi, Xu = Xt[inf_idx], Xt[unlabelled_idx]
    yi = y[inf_idx]
    rows, diagnostics, warnings = [], [], []

    # Naive ML.
    est, se = naive_ml_inference(task, Xu, pred_u)
    rows += _rows_for(
        "naive_ml", METHOD_LABELS["naive_ml"], est, se, truth, data["parameter_names"],
        data["experiment"], replicate, data["target_name"], info, quality, sizes=sizes,
    )

    # Standard PPI.
    ppi_est, ppi_se, _ = ppi_inference(task, Xi, yi, pred_inf, Xu, pred_u)
    rows += _rows_for(
        "ppi", METHOD_LABELS["ppi"], ppi_est, ppi_se, truth, data["parameter_names"],
        data["experiment"], replicate, data["target_name"], info, quality, sizes=sizes,
    )

    # PPI++V1, strict package wrapper.
    try:
        v1 = ppiplusv1_inference(task, Xi, yi, pred_inf, Xu, pred_u)
        rows += _rows_for(
            "ppi_plus_plus_v1", METHOD_LABELS["ppi_plus_plus_v1"], v1["estimate"], v1["se"],
            truth, data["parameter_names"], data["experiment"], replicate, data["target_name"],
            info, quality, selected_lambda=v1["selected_lambda"], lambda_source=v1["lambda_source"],
            backend=v1["backend"], package_version=v1["package_version"], sizes=sizes,
        )
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        rows += _failed_rows("ppi_plus_plus_v1", data["experiment"], replicate, data, truth, info, quality, msg, sizes)
        warnings.append(
            {
                "experiment": data["experiment"], "replicate": replicate, "learner": info["learner"],
                "method": "ppi_plus_plus_v1", "parameter": "all", "confidence_level": "all",
                "warning": msg, "traceback": traceback.format_exc(),
            }
        )

    # PPI++V2 internal grid.
    v2 = ppiplusv2_inference(task, Xi, yi, pred_inf, Xu, pred_u)
    rows += _rows_for(
        "ppi_plus_plus_v2", METHOD_LABELS["ppi_plus_plus_v2"], v2["estimate"], v2["se"],
        truth, data["parameter_names"], data["experiment"], replicate, data["target_name"], info,
        quality, selected_lambda=v2["selected_lambda"], lambda_source="internal_grid_trace", sizes=sizes,
    )
    for d in v2["grid_diagnostics"]:
        diagnostics.append(
            {
                "experiment": data["experiment"], "replicate": replicate, "learner": info["learner"],
                "method": "ppi_plus_plus_v2", "diagnostic": "lambda_grid", "lambda": d["lambda"],
                "value": d["cov_trace"], "fold_id": np.nan, "details": "",
            }
        )
    split_classic_est = classic_estimate_se(task, Xi, yi)[0]
    diagnostics.extend(
        [
            {
                "experiment": data["experiment"], "replicate": replicate, "learner": info["learner"],
                "method": "ppi_plus_plus_v2", "diagnostic": "lambda0_vs_split_classic_estimate_max_abs",
                "lambda": 0.0, "value": float(np.max(np.abs(v2["lambda0_estimate"] - split_classic_est))),
                "fold_id": np.nan, "details": "",
            },
            {
                "experiment": data["experiment"], "replicate": replicate, "learner": info["learner"],
                "method": "ppi_plus_plus_v2", "diagnostic": "lambda1_vs_ppi_estimate_max_abs",
                "lambda": 1.0, "value": float(np.max(np.abs(v2["lambda1_estimate"] - ppi_est))),
                "fold_id": np.nan, "details": "",
            },
        ]
    )

    # Cross-PPI fold diagnostics and strict package inference.
    for fd in cross["fold_details"]:
        diagnostics.append(
            {
                "experiment": data["experiment"], "replicate": replicate, "learner": info["learner"],
                "method": "cross_ppi", "diagnostic": "fold_tuning", "lambda": np.nan,
                "value": fd["validation_score"], "fold_id": fd["fold_id"],
                "details": json.dumps(fd["tuned_params"], sort_keys=True),
            }
        )
    diagnostics.append(
        {
            "experiment": data["experiment"], "replicate": replicate, "learner": info["learner"],
            "method": "cross_ppi", "diagnostic": "cross_oof_quality", "lambda": np.nan,
            "value": artifacts["cross_quality"], "fold_id": np.nan, "details": quality[0],
        }
    )
    try:
        ci = crossppi_inference(
            task,
            Xt[labelled_idx],
            y[labelled_idx],
            cross["pred_oof"],
            Xu,
            cross["pred_u_matrix"],
        )
        rows += _rows_for(
            "cross_ppi", METHOD_LABELS["cross_ppi"], ci["estimate"], ci["se"], truth,
            data["parameter_names"], data["experiment"], replicate, data["target_name"], info,
            quality, backend=ci["backend"], package_version=ci["package_version"], sizes=sizes,
        )
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        rows += _failed_rows("cross_ppi", data["experiment"], replicate, data, truth, info, quality, msg, sizes)
        warnings.append(
            {
                "experiment": data["experiment"], "replicate": replicate, "learner": info["learner"],
                "method": "cross_ppi", "parameter": "all", "confidence_level": "all",
                "warning": msg, "traceback": traceback.format_exc(),
            }
        )

    quality_row = {
        "experiment": data["experiment"],
        "replicate": replicate,
        "learner": info["learner"],
        "learner_label": info["learner_label"],
        "learner_model": info["learner_model"],
        "quality_metric": quality[0],
        "quality_value": quality[1],
        "tuned_params": json.dumps(info["tuned_params"], sort_keys=True),
        "validation_score": info["validation_score"],
        "n_train": len(train_idx),
        "n_inference": len(inf_idx),
    }
    return rows, diagnostics, warnings, quality_row


def run_experiment(mode, experiment):
    start = time.time()
    _, figdir, tabledir, otherdir = _prepare_dirs(mode, experiment)
    data = load_experiment(experiment)
    check_learner_ids(LEARNER_IDS)
    y, Xt = data["y"], data["X_target"]
    truth = compute_full_truth(data["task"], Xt, y)
    all_rows, all_diag, all_warn, all_quality, log = [], [], [], [], []
    B = MODES[mode]
    log.append(f"experiment={experiment} mode={mode} replicates={B} n={len(y)} truth={truth.tolist()}")

    for replicate in range(B):
        seed = RANDOM_SEED + 10000 * EXPERIMENTS.index(experiment) + replicate
        labelled_idx, unlabelled_idx = make_label_unlabel_split(len(y), seed)
        train_idx, inf_idx = make_ppi_train_inf_split(labelled_idx, y, data["binary"], seed + 1)
        folds = make_crossppi_folds(labelled_idx, y, data["binary"], seed + 2)
        sizes = {
            "n_total": len(y),
            "n_labelled": len(labelled_idx),
            "n_unlabelled": len(unlabelled_idx),
            "n_train": len(train_idx),
            "n_inference": len(inf_idx),
        }

        # Classic is learner-independent.
        est, se = classic_inference(data["task"], Xt[labelled_idx], y[labelled_idx])
        all_rows += _rows_for(
            "classic", METHOD_LABELS["classic"], est, se, truth, data["parameter_names"],
            experiment, replicate, data["target_name"], {}, None, sizes=sizes,
        )

        # Parallel stage: model fitting and prediction only.
        artifacts_by_learner = {}
        with ThreadPoolExecutor(max_workers=max(1, int(WORKER))) as pool:
            futures = {
                pool.submit(
                    _prepare_learner_artifacts,
                    lid,
                    data,
                    replicate,
                    seed,
                    labelled_idx,
                    unlabelled_idx,
                    train_idx,
                    inf_idx,
                    folds,
                ): lid
                for lid in LEARNER_IDS
            }
            for future in as_completed(futures):
                lid = futures[future]
                try:
                    artifacts_by_learner[lid] = future.result()
                except Exception as exc:
                    msg = f"Fatal learner preparation failure {lid}: {type(exc).__name__}: {exc}"
                    all_warn.append(
                        {
                            "experiment": experiment, "replicate": replicate, "learner": lid,
                            "method": "all", "parameter": "all", "confidence_level": "all",
                            "warning": msg, "traceback": traceback.format_exc(),
                        }
                    )

        # Serial stage: all inference, especially package calls and V2 grids.
        for lid in LEARNER_IDS:
            artifacts = artifacts_by_learner.get(lid)
            if artifacts is None:
                info = {
                    "learner": lid,
                    "learner_label": LEARNER_LABELS[lid],
                    "learner_model": model_name(lid, data["binary"]),
                    "tuned_params": {},
                    "validation_score": np.nan,
                }
                quality = (data["quality_metric"], np.nan)
                for method in ("naive_ml", "ppi", "ppi_plus_plus_v1", "ppi_plus_plus_v2", "cross_ppi"):
                    all_rows += _failed_rows(
                        method, experiment, replicate, data, truth, info, quality,
                        "Learner preparation failed; see warnings.csv", sizes,
                    )
                continue
            rows, diag, warn, q = _infer_from_artifacts(
                artifacts, data, replicate, labelled_idx, unlabelled_idx, train_idx, inf_idx, truth, sizes
            )
            all_rows += rows
            all_diag += diag
            all_warn += warn
            all_quality.append(q)

        log.append(f"replicate={replicate} completed elapsed={time.time() - start:.1f}s")

    rep, summary, main = aggregate_results(all_rows)
    display_parameters = set(data["display_parameters"])
    summary_display = summary[summary.parameter.isin(display_parameters)].copy()
    main_display = main[main.parameter.isin(display_parameters)].copy()

    rep = rep.sort_values(["experiment", "replicate", "learner", "method", "parameter", "confidence_level"], na_position="last")
    summary = summary.sort_values(["parameter", "learner", "method", "confidence_level"], na_position="last")
    summary_display = summary_display.sort_values(["parameter", "learner", "method", "confidence_level"], na_position="last")
    main_display = main_display.sort_values(["parameter", "learner", "method"], na_position="last")

    rep.to_csv(tabledir / "replicate_results.csv", index=False)
    summary.to_csv(tabledir / "summary_by_confidence.csv", index=False)
    main_display.to_csv(tabledir / "summary_95.csv", index=False)
    quality_df = pd.DataFrame(all_quality).sort_values(["replicate", "learner"])
    quality_df.to_csv(tabledir / "learner_quality.csv", index=False)
    training_summary = (quality_df.groupby(
        ["experiment", "learner", "learner_label", "learner_model", "tuned_params", "quality_metric"],
        as_index=False, dropna=False,
    ).agg(
        selection_count=("replicate", "size"),
        validation_score_mean=("validation_score", "mean"),
        validation_score_sd=("validation_score", "std"),
        quality_mean=("quality_value", "mean"),
        quality_sd=("quality_value", "std"),
        n_train=("n_train", "first"),
        n_inference=("n_inference", "first"),
    ))
    training_summary["selection_rate"] = training_summary["selection_count"] / training_summary.groupby(
        ["experiment", "learner"]
    )["selection_count"].transform("sum")
    training_summary = training_summary[
        ["experiment", "learner", "learner_label", "learner_model", "tuned_params",
         "selection_count", "selection_rate", "validation_score_mean", "validation_score_sd",
         "quality_metric", "quality_mean", "quality_sd", "n_train", "n_inference"]
    ]
    training_summary.to_csv(tabledir / "learner_training_summary.csv", index=False)
    diagnostics_df = pd.DataFrame(
        all_diag,
        columns=["experiment", "replicate", "learner", "method", "diagnostic", "lambda", "value", "fold_id", "details"],
    )
    diagnostics_df.to_csv(otherdir / "diagnostics.csv", index=False)
    tuning_summary = summarize_ppiv2_tuning_effect(rep, diagnostics_df)
    tuning_summary[tuning_summary.parameter.isin(data["display_parameters"])].to_csv(
        tabledir / "ppiv2_tuning_summary.csv", index=False
    )
    pd.DataFrame(all_warn, columns=WARNING_COLUMNS).to_csv(otherdir / "warnings.csv", index=False)

    config_snapshot = {
        "mode": mode,
        "replicates": B,
        "worker": WORKER,
        "parallel_stage": "learner fitting and prediction only",
        "inference_stage": "serial",
        "random_seed": RANDOM_SEED,
        "labelled_ratio": LABELLED_RATIO,
        "ppi_train_ratio": PPI_TRAIN_RATIO,
        "crossppi_k": CROSSPPI_K,
        "confidence_levels": CONFIDENCE_LEVELS,
        "lambda_grid": PPI_PLUS_PLUS_LAMBDA_GRID,
        "learner_ids": LEARNER_IDS,
        "learner_grids": LEARNER_GRIDS,
        "ppi_python_required": PPI_PY_REQUIRED_VERSION,
        "ppi_python_installed": _package_version(),
        "source_path": data["source_path"],
        "n_total": len(y),
        "truth": truth.tolist(),
    }
    (otherdir / "config_used.json").write_text(json.dumps(config_snapshot, indent=2, default=str), encoding="utf-8")
    (otherdir / "run_log.txt").write_text(
        "\n".join(log) + f"\nfinished elapsed={time.time() - start:.1f}s\n", encoding="utf-8"
    )
    plot_learner_quality(quality_df, figdir, experiment, B)
    plot_inference_performance_95(main_display, quality_df, figdir, experiment, B)
    plot_coverage_calibration_by_parameter(summary_display, figdir, experiment, data["display_parameters"], B)
    print(f"Completed {experiment} ({mode}) in {time.time() - start:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--experiment", choices=EXPERIMENTS)
    args = parser.parse_args()
    experiments = (args.experiment,) if args.experiment else EXPERIMENTS
    for exp in experiments:
        run_experiment(args.mode, exp)


if __name__ == "__main__":
    main()
