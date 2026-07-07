"""Central configuration for the controlled PPI toy experiment.

All numerical settings belong here.  The command line only selects the run
mode (fast or full); it does not alter the experiment definition.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# ---------------------------------------------------------------------------
# Experiment identity and active scope
# ---------------------------------------------------------------------------
PROJECT_NAME = "toy"
ACTIVE_SCENARIOS = ("mean", "lr", "logistic_glm")
ACTIVE_METHODS = ("classic", "naive_ml", "ppi", "ppi_plus_plus")
ACTIVE_PROFILES = ("P1", "P2", "P3", "P4")
CROSS_PPI_ENABLED = False
CROSS_PPI_K = 5  # Reserved for the later Cross-PPI extension.

# ---------------------------------------------------------------------------
# Data-generating mechanism
# ---------------------------------------------------------------------------
N_LABELED = 60
N_UNLABELED = 600
X1_LOW = -1.0
X1_HIGH = 1.0
X2_PROBABILITY = 0.5
TRUE_BETA = (0.25, 0.90, -0.60)
GAUSSIAN_NOISE_SD = 1.0

# ---------------------------------------------------------------------------
# Controlled proxy-output regimes
# P1/P4 use externally generated U ~ Uniform(-1, 1).  U is part of the
# controlled proxy-output construction, not an inferential covariate.
# ---------------------------------------------------------------------------
P1_NOISE_AMPLITUDE = 0.02
P2_CONSTANT_SHIFT = 0.06
P3_SIGNED_SHIFT = 0.06
P4_NOISE_AMPLITUDE = 0.20

# ---------------------------------------------------------------------------
# Inference and numerical settings
# ---------------------------------------------------------------------------
CONFIDENCE_LEVELS = (0.90, 0.95, 0.975)
PPI_PLUS_PLUS_LAMBDA_MODE = "min_sandwich_trace"
PPI_PLUS_PLUS_FIXED_LAMBDA = None
PPI_PLUS_PLUS_LAMBDA_GRID = tuple(round(i * 0.05, 2) for i in range(21))

GLM_MAX_ITER = 100
GLM_TOL = 1e-10
GLM_LINESEARCH_MAX_STEPS = 25
MAX_HESSIAN_CONDITION_NUMBER = 1e12
FAIL_ON_NONCONVERGENCE = True

# ---------------------------------------------------------------------------
# Execution settings
# ---------------------------------------------------------------------------
# Number of independent replicate processes.  Set to 1 for serial execution;
# use a positive integer no larger than the CPU capacity you want to allocate.
# Random-number streams are keyed by (seed, replicate_id, stream_id), so this
# setting changes execution speed only, not the simulated data or estimates.
WORKERS = 13

MAIN_SEED = 0
ROBUSTNESS_SEEDS = tuple(range(1, 30))
PROGRESS_EVERY_FAST = 25
PROGRESS_EVERY_FULL = 100

RUN_CONFIGS = {
    "fast": {
        "n_replicates": 200,
        "seeds": (MAIN_SEED,),
        "save_input_data": "all",
        "save_replicate_results": "all",
        "progress_every": PROGRESS_EVERY_FAST,
    },
    "full": {
        "n_replicates": 2000,
        "seeds": (MAIN_SEED,) + ROBUSTNESS_SEEDS,
        "save_input_data": "seed0_only",
        "save_replicate_results": "seed0_only",
        "progress_every": PROGRESS_EVERY_FULL,
    },
}

# ---------------------------------------------------------------------------
# Output and plotting settings
# ---------------------------------------------------------------------------
SAVE_FIGURE_FORMATS = ("png", "pdf")
FIGURE_DPI = 220
PLOT_COVERAGE_YLIM = (0.0, 1.0)
# Main text reports the nominal 95% interval.  Other configured levels are
# retained in appendix calibration figures.
MAIN_FIGURE_CONFIDENCE_LEVEL = 0.95
# Calibration figures display coverage minus nominal coverage on this shared
# scale.  Catastrophic Naive-ML failures outside the scale are marked with
# censored triangles so calibrated methods remain readable.
CALIBRATION_COVERAGE_ERROR_YLIM = (-0.12, 0.12)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_run_config(mode: str) -> dict[str, Any]:
    """Return a detached run-mode configuration."""
    if mode not in RUN_CONFIGS:
        raise ValueError(f"Unknown mode {mode!r}. Choose one of {tuple(RUN_CONFIGS)}.")
    return deepcopy(RUN_CONFIGS[mode])


def snapshot() -> dict[str, Any]:
    """Return a JSON-serialisable snapshot of uppercase configuration values."""
    values: dict[str, Any] = {}
    for key, value in globals().items():
        if key.isupper() and not key.startswith("__"):
            if isinstance(value, tuple):
                values[key] = list(value)
            elif isinstance(value, dict):
                values[key] = {
                    str(k): (list(v) if isinstance(v, tuple) else v)
                    for k, v in value.items()
                }
            else:
                values[key] = value
    return values
