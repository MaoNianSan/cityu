from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from config import EXPERIMENTS, MODES, OUTPUT_DIR, DATASET_CONFIG
from metrics import summarize_ppiv2_tuning_effect
from plot import (
    plot_learner_quality,
    plot_inference_performance_95,
    plot_coverage_calibration_by_parameter,
)

OLD_STEMS = (
    "coverage_width_95",
    "coverage_calibration",
    "naive_ml_coverage_diagnostic",
    "ppiv2_lambda_distribution",
    "inference_performance_95",
    "learner_quality",
)


def replot(mode, experiment):
    base = OUTPUT_DIR / mode / experiment
    fig = base / "figure"
    tab = base / "table"
    other = base / "other"
    paths = {
        "s95": tab / "summary_95.csv",
        "summary": tab / "summary_by_confidence.csv",
        "quality": tab / "learner_quality.csv",
        "rep": tab / "replicate_results.csv",
        "diag": other / "diagnostics.csv",
        "config": other / "config_used.json",
    }
    missing = [str(x) for x in paths.values() if not x.exists()]
    if missing:
        raise FileNotFoundError("Missing existing output files:\n" + "\n".join(missing))
    for stem in OLD_STEMS:
        for ext in ("png", "pdf"):
            (fig / f"{stem}.{ext}").unlink(missing_ok=True)
    for p in fig.glob("diagnostic_coverage_calibration_*.png"):
        p.unlink()
    for p in fig.glob("diagnostic_coverage_calibration_*.pdf"):
        p.unlink()
    (tab / "plot_summary.csv").unlink(missing_ok=True)
    s95 = pd.read_csv(paths["s95"])
    summary = pd.read_csv(paths["summary"])
    q = pd.read_csv(paths["quality"])
    rep = pd.read_csv(paths["rep"])
    diag = pd.read_csv(paths["diag"])
    cfg = json.loads(paths["config"].read_text(encoding="utf-8"))
    B = int(cfg.get("replicates", rep.replicate.nunique()))
    plot_learner_quality(q, fig, experiment, B)
    plot_inference_performance_95(s95, q, fig, experiment, B)
    plot_coverage_calibration_by_parameter(
        summary,
        fig,
        experiment,
        DATASET_CONFIG[experiment].get(
            "display_parameters", DATASET_CONFIG[experiment]["parameters"]
        ),
        B,
    )
    main_parameters = DATASET_CONFIG[experiment].get(
        "display_parameters", DATASET_CONFIG[experiment]["parameters"]
    )
    tuning = summarize_ppiv2_tuning_effect(rep, diag)
    tuning[tuning.parameter.isin(main_parameters)].to_csv(
        tab / "ppiv2_tuning_summary.csv", index=False
    )
    print(f"Replotted {experiment} ({mode}) from existing outputs.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--experiment", choices=EXPERIMENTS)
    a = ap.parse_args()
    for e in ((a.experiment,) if a.experiment else EXPERIMENTS):
        replot(a.mode, e)


if __name__ == "__main__":
    main()
