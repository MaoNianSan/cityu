from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"

EXPERIMENTS = ("a_mean", "b_lr", "c_glm")
MODES = {"fast": 20, "full": 100}
RANDOM_SEED = 20260709
WORKER = 4  # for 4 learners, so the max spped is 4

LABELLED_RATIO = 0.10  # unmasked data ratio
PPI_TRAIN_RATIO = 0.20  #
CROSSPPI_K = 5
TUNING_VALIDATION_RATIO = 0.25
CONFIDENCE_LEVELS = (0.90, 0.95, 0.975)  # CI
MAIN_CONFIDENCE_LEVEL = 0.95
PPI_PLUS_PLUS_LAMBDA_GRID = tuple(round(i * 0.025, 3) for i in range(41))
# lambda solution PPI++V2
PROBABILITY_EPS = 1e-6
CLEAN_OUTPUT_BEFORE_RUN = True
PPI_PY_REQUIRED_VERSION = "0.2.3"

EXPERIMENT_LABELS = {
    "a_mean": "Galaxy Zoo 2 mean estimation",
    "b_lr": "ACS PUMS linear regression",
    "c_glm": "Adult logistic GLM",
}
PARAMETER_LABELS = {
    "smooth_proportion": "Smooth-galaxy proportion",
    "intercept": "Intercept",
    "AGEP": "Age coefficient",
    "SEX_MALE": "Male-indicator coefficient",
    "age": "Age coefficient",
    "education-num": "Education-years coefficient",
    "hours-per-week": "Hours-per-week coefficient",
}

LEARNER_IDS = ("lin", "dt", "rf", "gb")
LEARNER_LABELS = {"lin": "LIN", "dt": "DT", "rf": "RF", "gb": "GB"}
METHOD_LABELS = {
    "classic": "Classic",
    "naive_ml": "Naive ML",
    "ppi": "PPI",
    "ppi_plus_plus_v1": "PPI++V1",
    "ppi_plus_plus_v2": "PPI++V2",
    "cross_ppi": "Cross-PPI",
}
METHOD_ORDER = tuple(METHOD_LABELS)


LEARNER_GRIDS = {
    "lin_binary": [
        {"model__C": 0.1},
        {"model__C": 1.0},
        {"model__C": 10.0},
    ],
    "lin_continuous": [
        {"model__alpha": 0.1},
        {"model__alpha": 1.0},
        {"model__alpha": 10.0},
    ],
    "dt": [
        {"model__max_depth": 4, "model__min_samples_leaf": 10},
        {"model__max_depth": 4, "model__min_samples_leaf": 50},
        {"model__max_depth": 8, "model__min_samples_leaf": 10},
        {"model__max_depth": 8, "model__min_samples_leaf": 50},
    ],
    "rf": [
        {
            "model__n_estimators": 20,
            "model__max_depth": 8,
            "model__min_samples_leaf": 20,
        },
        {
            "model__n_estimators": 20,
            "model__max_depth": 12,
            "model__min_samples_leaf": 20,
        },
    ],
    "gb": [
        {
            "model__max_iter": 50,
            "model__learning_rate": 0.1,
            "model__max_leaf_nodes": 15,
        },
        {
            "model__max_iter": 100,
            "model__learning_rate": 0.1,
            "model__max_leaf_nodes": 15,
        },
    ],
}

DATASET_CONFIG = {
    "a_mean": {
        "path": INPUT_DIR / "galaxy_zoo_2" / "gz2_hart16.csv",
        "task": "mean",
        "target_name": "smooth_proportion",
        "outcome": "t01_smooth_or_features_a01_smooth_flag",
        "learner_features": [
            "ra",
            "dec",
            "sample",
            "total_classifications",
            "total_votes",
        ],
        "numeric_features": ["ra", "dec", "total_classifications", "total_votes"],
        "categorical_features": ["sample"],
        "target_features": [],
        "parameters": ["smooth_proportion"],
        "quality_metric": "Brier",
        "binary": True,
    },
    "b_lr": {
        "path": INPUT_DIR / "acs_pums" / "acs_pums_ca_2019_person.csv",
        "task": "linear",
        "target_name": "income_on_age_sex",
        "outcome": "PINCP",
        "learner_features": [
            "AGEP",
            "SEX_MALE",
            "SCHL",
            "MAR",
            "COW",
            "WKHP",
            "POBP",
            "RAC1P",
            "HICOV",
            "ESR",
        ],
        "numeric_features": ["AGEP", "WKHP"],
        "categorical_features": [
            "SEX_MALE",
            "SCHL",
            "MAR",
            "COW",
            "POBP",
            "RAC1P",
            "HICOV",
            "ESR",
        ],
        "target_features": ["AGEP", "SEX_MALE"],
        "parameters": ["intercept", "AGEP", "SEX_MALE"],
        "display_parameters": ["AGEP", "SEX_MALE"],
        "quality_metric": "MSE",
        "binary": False,
    },
    "c_glm": {
        "path": INPUT_DIR / "adult" / "adult_reconstruction.csv",
        "task": "logistic",
        "target_name": "income_threshold_logistic",
        "outcome": "income",
        "learner_features": [
            "hours-per-week",
            "age",
            "capital-gain",
            "capital-loss",
            "workclass",
            "education",
            "education-num",
            "marital-status",
            "relationship",
            "race",
            "gender",
            "native-country",
            "occupation",
        ],
        "numeric_features": [
            "hours-per-week",
            "age",
            "capital-gain",
            "capital-loss",
            "education-num",
        ],
        "categorical_features": [
            "workclass",
            "education",
            "marital-status",
            "relationship",
            "race",
            "gender",
            "native-country",
            "occupation",
        ],
        "target_features": ["age", "education-num", "hours-per-week"],
        "parameters": ["intercept", "age", "education-num", "hours-per-week"],
        "display_parameters": ["age", "education-num", "hours-per-week"],
        "quality_metric": "Brier",
        "binary": True,
    },
}
