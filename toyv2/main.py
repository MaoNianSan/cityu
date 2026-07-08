from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm

import config
from baselines import fit_classic, fit_naive_ml
from checks import run_preflight_checks
from data_generation import generate_replicate, save_replicate
from formulation import EstimatorResult, ScenarioSpec, all_scenarios
from learner_proxy import generate_proxy
from plotting import plot_all_results
from ppi import fit_ppi
from ppiplusplus import fit_ppi_plus_plus

ROOT = Path(__file__).resolve().parent


@dataclass
class MetricAccumulator:
    """Online aggregation that avoids storing non-seed0 replicate records."""

    storage: dict

    def __init__(self) -> None:
        self.storage = defaultdict(
            lambda: {"count": 0, "sum_width": 0.0, "sum_coverage": 0.0}
        )

    def add(self, key: tuple, width: float, covered: bool) -> None:
        bucket = self.storage[key]
        bucket["count"] += 1
        bucket["sum_width"] += float(width)
        bucket["sum_coverage"] += float(covered)

    def to_frame(self) -> pd.DataFrame:
        rows: list[dict] = []
        for key, values in self.storage.items():
            (
                seed,
                scenario,
                target,
                profile,
                method,
                confidence_level,
            ) = key
            count = values["count"]
            rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "target": target,
                    "profile": profile,
                    "method": method,
                    "confidence_level": confidence_level,
                    "n_labeled": config.N_LABELED,
                    "n_unlabeled": config.N_UNLABELED,
                    "n_replicates": count,
                    "average_ci_width": values["sum_width"] / count,
                    "empirical_coverage": values["sum_coverage"] / count,
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values(
                ["seed", "scenario", "target", "confidence_level", "profile", "method"]
            )
            .reset_index(drop=True)
        )


def _setup_logger(mode: str) -> logging.Logger:
    other_dir = ROOT / "output" / mode / "other"
    other_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"toy.{mode}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(
        other_dir / "run_log.txt", mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def _prepare_directories(mode: str) -> dict[str, Path]:
    output_root = ROOT / "output" / mode
    input_root = ROOT / "input" / mode
    if mode == "full":
        input_data_root = input_root / "seed0" / "data"
    else:
        input_data_root = input_root / "data"
    paths = {
        "output_root": output_root,
        "figure": output_root / "figure",
        "table": output_root / "table",
        "other": output_root / "other",
        "input_root": input_root,
        "input_data": input_data_root,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _should_save(setting: str, seed: int) -> bool:
    if setting == "all":
        return True
    if setting == "seed0_only":
        return seed == config.MAIN_SEED
    if setting == "none":
        return False
    raise ValueError(f"Unknown save setting {setting!r}.")


def _critical_value(confidence_level: float) -> float:
    return float(norm.ppf(0.5 + confidence_level / 2.0))


def _interval_covers(lower: float, upper: float, target: float) -> bool:
    """Evaluate interval coverage while absorbing floating-point boundary noise.

    This matters when a deterministic pseudo outcome yields a numerically zero
    standard error and the target equals an interval endpoint analytically.
    The tolerance is bounded at floating-point round-off scale and does not
    alter substantive interval comparisons.
    """
    scale = max(1.0, abs(lower), abs(upper), abs(target))
    tolerance = 64.0 * np.finfo(float).eps * scale
    return bool(lower - tolerance <= target <= upper + tolerance)


def _diagnostic_row(
    seed: int,
    replicate_id: int,
    scenario: ScenarioSpec,
    profile: str,
    method: str,
    result: EstimatorResult,
) -> dict:
    diagnostics = result.diagnostics
    candidate_traces = diagnostics.get("lambda_candidate_traces")
    return {
        "seed": seed,
        "replicate_id": replicate_id,
        "scenario": scenario.name,
        "profile": profile,
        "method": method,
        "converged": bool(result.converged),
        "lambda_hat": diagnostics.get("lambda_hat"),
        "iterations": diagnostics.get("iterations"),
        "condition_number": diagnostics.get("condition_number"),
        "gradient_max_abs": diagnostics.get("gradient_max_abs"),
        "selected_covariance_trace": diagnostics.get("selected_covariance_trace"),
        "lambda_selection_mode": diagnostics.get("lambda_selection_mode"),
        "lambda_candidate_traces": (
            json.dumps(candidate_traces) if candidate_traces is not None else None
        ),
        "failure_reason": diagnostics.get("failure_reason"),
    }


def _record_result(
    accumulator: MetricAccumulator,
    replicate_records: list[dict] | None,
    diagnostics: list[dict] | None,
    seed: int,
    replicate_id: int,
    scenario: ScenarioSpec,
    profile: str,
    method: str,
    result: EstimatorResult,
) -> None:
    if diagnostics is not None:
        diagnostics.append(
            _diagnostic_row(seed, replicate_id, scenario, profile, method, result)
        )

    if not result.converged:
        message = (
            f"Estimator failed: seed={seed}, replicate={replicate_id}, "
            f"scenario={scenario.name}, profile={profile}, method={method}, "
            f"reason={result.diagnostics.get('failure_reason')}"
        )
        if config.FAIL_ON_NONCONVERGENCE:
            raise RuntimeError(message)
        return

    estimates = np.asarray(result.estimate, dtype=float)
    covariance = np.asarray(result.covariance, dtype=float)
    if not (np.all(np.isfinite(estimates)) and np.all(np.isfinite(covariance))):
        raise RuntimeError(
            "A converged estimator returned non-finite estimates or covariance."
        )

    for index, target in enumerate(scenario.target_names):
        variance = float(covariance[index, index])
        if variance < -1e-12:
            raise RuntimeError(
                f"Negative variance for {scenario.name}/{target}: {variance}."
            )
        standard_error = float(np.sqrt(max(variance, 0.0)))
        target_value = float(scenario.true_values[index])
        estimate = float(estimates[index])

        for confidence_level in config.CONFIDENCE_LEVELS:
            critical_value = _critical_value(confidence_level)
            width = 2.0 * critical_value * standard_error
            lower = estimate - critical_value * standard_error
            upper = estimate + critical_value * standard_error
            covered = _interval_covers(lower, upper, target_value)
            key = (seed, scenario.name, target, profile, method, confidence_level)
            accumulator.add(key, width, covered)

            if replicate_records is not None:
                replicate_records.append(
                    {
                        "seed": seed,
                        "replicate_id": replicate_id,
                        "scenario": scenario.name,
                        "target": target,
                        "profile": profile,
                        "method": method,
                        "confidence_level": confidence_level,
                        "estimate": estimate,
                        "standard_error": standard_error,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "ci_width": width,
                        "covered": covered,
                        "true_value": target_value,
                        "lambda_hat": result.diagnostics.get("lambda_hat"),
                    }
                )


def _evaluate_replicate(
    seed: int,
    replicate_id: int,
    save_input: bool,
    input_data_dir: Path | None,
) -> tuple[int, list[tuple[str, str, str, EstimatorResult]], dict | None]:
    """Evaluate one replicate without side effects other than its optional input cache.

    The function is module-level so it can be executed by ``ProcessPoolExecutor``
    on Windows as well as Unix-like systems.  Its random streams are fully keyed
    by ``(seed, replicate_id, stream_id)`` in ``data_generation.py``.
    """
    data = generate_replicate(seed=seed, replicate_id=replicate_id)
    input_manifest_row: dict | None = None
    if save_input:
        if input_data_dir is None:
            raise ValueError("input_data_dir is required when save_input=True.")
        seed_dir = input_data_dir / f"seed_{seed:02d}"
        file_path = seed_dir / f"replicate_{replicate_id:05d}.npz"
        save_replicate(data, file_path)
        input_manifest_row = {
            "seed": seed,
            "replicate_id": replicate_id,
            "file_path": file_path,
        }

    results: list[tuple[str, str, str, EstimatorResult]] = []
    for scenario in all_scenarios():
        results.append(
            (scenario.name, "baseline", "classic", fit_classic(scenario, data))
        )
        for profile in config.ACTIVE_PROFILES:
            prediction = generate_proxy(data, scenario, profile)
            results.extend(
                [
                    (
                        scenario.name,
                        profile,
                        "naive_ml",
                        fit_naive_ml(scenario, data, prediction),
                    ),
                    (
                        scenario.name,
                        profile,
                        "ppi",
                        fit_ppi(scenario, data, prediction),
                    ),
                    (
                        scenario.name,
                        profile,
                        "ppi_plus_plus",
                        fit_ppi_plus_plus(scenario, data, prediction),
                    ),
                ]
            )
    return replicate_id, results, input_manifest_row


def _run_seed(
    mode: str,
    seed: int,
    run_config: dict,
    paths: dict[str, Path],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, list[dict], list[dict], list[dict]]:
    accumulator = MetricAccumulator()
    collect_detailed = _should_save(run_config["save_replicate_results"], seed)
    collect_diagnostics = collect_detailed
    replicate_records: list[dict] | None = [] if collect_detailed else None
    diagnostics: list[dict] | None = [] if collect_diagnostics else None
    input_manifest: list[dict] = []
    save_inputs = _should_save(run_config["save_input_data"], seed)

    scenarios_by_name = {scenario.name: scenario for scenario in all_scenarios()}
    n_replicates = int(run_config["n_replicates"])
    progress_every = int(run_config["progress_every"])
    workers = int(config.WORKERS)
    if workers < 1:
        raise ValueError("config.WORKERS must be a positive integer.")

    evaluations: list[
        tuple[int, list[tuple[str, str, str, EstimatorResult]], dict | None] | None
    ] = [None] * n_replicates
    input_data_dir = paths["input_data"] if save_inputs else None

    if workers == 1:
        for replicate_id in range(n_replicates):
            evaluations[replicate_id] = _evaluate_replicate(
                seed=seed,
                replicate_id=replicate_id,
                save_input=save_inputs,
                input_data_dir=input_data_dir,
            )
            if (
                replicate_id + 1
            ) % progress_every == 0 or replicate_id + 1 == n_replicates:
                logger.info(
                    "mode=%s seed=%s replicate=%s/%s",
                    mode,
                    seed,
                    replicate_id + 1,
                    n_replicates,
                )
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_replicate,
                    seed,
                    replicate_id,
                    save_inputs,
                    input_data_dir,
                ): replicate_id
                for replicate_id in range(n_replicates)
            }
            for future in as_completed(futures):
                replicate_id = futures[future]
                evaluations[replicate_id] = future.result()
                completed += 1
                if completed % progress_every == 0 or completed == n_replicates:
                    logger.info(
                        "mode=%s seed=%s completed=%s/%s",
                        mode,
                        seed,
                        completed,
                        n_replicates,
                    )

    # Aggregate in replicate-id order, preserving numerical and table-order
    # reproducibility between serial and parallel execution.
    for replicate_id, evaluation in enumerate(evaluations):
        if evaluation is None:
            raise RuntimeError(
                f"Missing completed evaluation for replicate {replicate_id}."
            )
        returned_id, results, input_row = evaluation
        if returned_id != replicate_id:
            raise RuntimeError(
                "Replicate evaluation was returned with an inconsistent identifier."
            )
        if input_row is not None:
            file_path = Path(input_row["file_path"])
            input_manifest.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "replicate_id": replicate_id,
                    "file": file_path.relative_to(ROOT).as_posix(),
                    "n_labeled": config.N_LABELED,
                    "n_unlabeled": config.N_UNLABELED,
                }
            )

        for scenario_name, profile, method, result in results:
            _record_result(
                accumulator,
                replicate_records,
                diagnostics,
                seed,
                replicate_id,
                scenarios_by_name[scenario_name],
                profile,
                method,
                result,
            )

    seed_metrics = accumulator.to_frame()
    return seed_metrics, (replicate_records or []), (diagnostics or []), input_manifest


def _robustness_summary(all_metrics: pd.DataFrame) -> pd.DataFrame:
    seed0 = all_metrics[all_metrics["seed"] == config.MAIN_SEED].copy()
    robustness = all_metrics[all_metrics["seed"].isin(config.ROBUSTNESS_SEEDS)].copy()
    key_columns = ["scenario", "target", "profile", "method", "confidence_level"]

    if robustness.empty:
        return pd.DataFrame()

    summary = robustness.groupby(key_columns, as_index=False).agg(
        average_ci_width_median_seed1_29=("average_ci_width", "median"),
        average_ci_width_q25_seed1_29=("average_ci_width", lambda x: x.quantile(0.25)),
        average_ci_width_q75_seed1_29=("average_ci_width", lambda x: x.quantile(0.75)),
        average_ci_width_min_seed1_29=("average_ci_width", "min"),
        average_ci_width_max_seed1_29=("average_ci_width", "max"),
        empirical_coverage_median_seed1_29=("empirical_coverage", "median"),
        empirical_coverage_q25_seed1_29=(
            "empirical_coverage",
            lambda x: x.quantile(0.25),
        ),
        empirical_coverage_q75_seed1_29=(
            "empirical_coverage",
            lambda x: x.quantile(0.75),
        ),
        empirical_coverage_min_seed1_29=("empirical_coverage", "min"),
        empirical_coverage_max_seed1_29=("empirical_coverage", "max"),
    )
    seed0_subset = seed0[
        key_columns + ["average_ci_width", "empirical_coverage"]
    ].rename(
        columns={
            "average_ci_width": "seed0_average_ci_width",
            "empirical_coverage": "seed0_empirical_coverage",
        }
    )
    return (
        seed0_subset.merge(summary, on=key_columns, how="left")
        .sort_values(key_columns)
        .reset_index(drop=True)
    )


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_detail_table(
    frame: pd.DataFrame, parquet_path: Path, logger: logging.Logger
) -> Path:
    """Write detailed audit records, falling back to compressed CSV if pyarrow is unavailable."""
    try:
        frame.to_parquet(parquet_path, index=False)
        return parquet_path
    except (ImportError, ModuleNotFoundError):
        fallback = parquet_path.with_suffix(".csv.gz")
        frame.to_csv(fallback, index=False, compression="gzip")
        logger.warning(
            "pyarrow is unavailable; wrote %s instead of %s.",
            fallback.name,
            parquet_path.name,
        )
        return fallback


def _write_manifest(mode: str, run_config: dict, path: Path) -> None:
    package_versions: dict[str, str] = {}
    for package in ("numpy", "pandas", "scipy", "matplotlib"):
        try:
            module = __import__(package)
            package_versions[package] = getattr(module, "__version__", "unknown")
        except Exception:
            package_versions[package] = "unavailable"
    payload = {
        "project": config.PROJECT_NAME,
        "mode": mode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions,
        "run_config": run_config,
        "config": config.snapshot(),
    }
    _write_json(path, payload)


def run(mode: str, skip_checks: bool = False, skip_plots: bool = False) -> None:
    if (
        isinstance(config.WORKERS, bool)
        or int(config.WORKERS) != config.WORKERS
        or int(config.WORKERS) < 1
    ):
        raise ValueError("config.WORKERS must be a positive integer.")
    if config.CROSS_PPI_ENABLED:
        raise RuntimeError(
            "Cross-PPI is reserved but not implemented in the current preliminary experiment."
        )
    run_config = config.get_run_config(mode)
    paths = _prepare_directories(mode)
    logger = _setup_logger(mode)

    _write_json(paths["other"] / "config_snapshot.json", config.snapshot())
    _write_manifest(mode, run_config, paths["other"] / "run_manifest.json")

    if not skip_checks:
        logger.info("Running preflight checks.")
        run_preflight_checks()
        logger.info("Preflight checks passed.")

    all_seed_metrics: list[pd.DataFrame] = []
    all_replicate_records: list[dict] = []
    all_diagnostics: list[dict] = []
    all_input_manifest: list[dict] = []

    for seed in run_config["seeds"]:
        logger.info("Starting mode=%s seed=%s.", mode, seed)
        seed_metrics, replicate_records, diagnostics, input_manifest = _run_seed(
            mode, int(seed), run_config, paths, logger
        )
        all_seed_metrics.append(seed_metrics)
        all_replicate_records.extend(replicate_records)
        all_diagnostics.extend(diagnostics)
        all_input_manifest.extend(input_manifest)

    metrics = pd.concat(all_seed_metrics, ignore_index=True)
    table_dir = paths["table"]
    other_dir = paths["other"]

    if mode == "fast":
        metrics.to_csv(table_dir / "metrics_seed0.csv", index=False)
        seed0_metrics = metrics[metrics["seed"] == config.MAIN_SEED].copy()
        robustness = None
    else:
        metrics.to_csv(table_dir / "all_seed_metrics.csv", index=False)
        seed0_metrics = metrics[metrics["seed"] == config.MAIN_SEED].copy()
        seed0_metrics.to_csv(table_dir / "seed0_metrics.csv", index=False)
        robustness = _robustness_summary(metrics)
        robustness.to_csv(table_dir / "robustness_summary.csv", index=False)

    if all_replicate_records:
        _write_detail_table(
            pd.DataFrame(all_replicate_records),
            other_dir / "replicate_results_seed0.parquet",
            logger,
        )
    if all_diagnostics:
        _write_detail_table(
            pd.DataFrame(all_diagnostics),
            other_dir / "diagnostics_seed0.parquet",
            logger,
        )
    if all_input_manifest:
        pd.DataFrame(all_input_manifest).to_csv(
            paths["input_root"] / "manifest.csv", index=False
        )

    if not skip_plots:
        plot_all_results(
            seed0_metrics,
            paths["figure"],
            robustness_summary=robustness,
            all_metrics=metrics,
        )
        logger.info("Figures written to %s.", paths["figure"])

    logger.info("Completed mode=%s.", mode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the controlled PPI toy experiment."
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "full"),
        required=True,
        help="Execution mode from config.py.",
    )
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip checks.py preflight assertions.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip figure generation after tables are written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        mode=arguments.mode,
        skip_checks=arguments.skip_checks,
        skip_plots=arguments.skip_plots,
    )
