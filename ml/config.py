"""Central configuration for the MultiDx ML pipeline.

Paths, reproducibility, the task registry, model zoo and tuning grids.
All randomness is controlled by :data:`SEED`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS_DIR = Path(os.environ.get("MULTIDX_ARTIFACTS", ROOT / "ml" / "artifacts"))
REPORTS_DIR = ROOT / "reports"

for _d in (DATA_RAW, DATA_PROCESSED, ARTIFACTS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility / evaluation
# --------------------------------------------------------------------------- #
SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
TUNING_ITER = 12  # RandomizedSearchCV candidates per model
CALIBRATION_FOLDS = 5

DISCLAIMER = "Preliminary model-based indication. Not a medical diagnosis."

# --------------------------------------------------------------------------- #
# Model zoo
# --------------------------------------------------------------------------- #
MODEL_NAMES = ["logreg", "random_forest", "xgboost", "lightgbm", "svm"]

MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "svm": "SVM",
}

# LR/SVM need feature scaling; trees do not.
NEEDS_SCALING = {
    "logreg": True,
    "random_forest": False,
    "xgboost": False,
    "lightgbm": False,
    "svm": True,
}

# --------------------------------------------------------------------------- #
# Tuning grids (pipeline step names: "pre" -> "clf")
# --------------------------------------------------------------------------- #
PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logreg": {
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "random_forest": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 8, 16],
        "clf__min_samples_leaf": [1, 3, 5],
        "clf__max_features": ["sqrt", 0.5],
    },
    "xgboost": {
        "clf__n_estimators": [200, 400],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__max_depth": [3, 5, 7],
        "clf__subsample": [0.8, 1.0],
    },
    "lightgbm": {
        "clf__n_estimators": [200, 400],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__num_leaves": [15, 31, 63],
        "clf__max_depth": [-1, 5, 10],
    },
    "svm": {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__kernel": ["rbf", "linear"],
        "clf__gamma": ["scale", 0.01],
    },
}

# --------------------------------------------------------------------------- #
# Risk / abstention configuration (used by the backend)
# --------------------------------------------------------------------------- #
RISK_CONFIG: dict[str, Any] = {
    "grey_zone": 0.05,
    "moderate_threshold": 0.33,
    "high_threshold": 0.66,
}

# --------------------------------------------------------------------------- #
# Task registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TaskSpec:
    """One prediction task."""

    task_id: str
    label: str
    kind: str  # "multiclass" | "binary"
    tier: int
    dataset: str
    short_label: str = ""
    # Fraction of model input features that must be present before predicting.
    min_coverage: float = 0.0
    required_inputs: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""


TASKS: dict[str, TaskSpec] = {
    "symptoms": TaskSpec(
        task_id="symptoms",
        label="Symptom-based differential (41 diseases)",
        short_label="Symptom differential",
        kind="multiclass",
        tier=1,
        dataset="Kaggle: Disease Symptom Prediction",
        min_coverage=0.0,
        notes="Multi-class; answered by a searchable symptom multi-select.",
    ),
    "heart": TaskSpec(
        task_id="heart",
        label="Heart disease (UCI Cleveland)",
        short_label="Heart disease",
        kind="binary",
        tier=2,
        dataset="UCI Heart Disease - processed.cleveland.data",
        min_coverage=0.50,
        required_inputs=("age", "sex", "trestbps"),
    ),
    "diabetes": TaskSpec(
        task_id="diabetes",
        label="Diabetes (Pima Indians)",
        short_label="Diabetes",
        kind="binary",
        tier=2,
        dataset="Pima Indians Diabetes Database (UCI/NIDDK mirror)",
        min_coverage=0.50,
        required_inputs=("age", "glucose"),
    ),
    "liver": TaskSpec(
        task_id="liver",
        label="Liver disease (ILPD)",
        short_label="Liver disease",
        kind="binary",
        tier=2,
        dataset="UCI Indian Liver Patient Dataset (Kaggle mirror)",
        min_coverage=0.50,
        required_inputs=("age", "sex"),
    ),
    "ckd": TaskSpec(
        task_id="ckd",
        label="Chronic kidney disease (UCI CKD)",
        short_label="Chronic kidney disease",
        kind="binary",
        tier=2,
        dataset="UCI Chronic Kidney Disease (arff)",
        min_coverage=0.50,
        required_inputs=("age", "bp"),
    ),
    "cancer": TaskSpec(
        task_id="cancer",
        label="Breast cancer malignancy (WDBC)",
        short_label="Breast cancer",
        kind="binary",
        tier=2,
        dataset="UCI Wisconsin Diagnostic Breast Cancer (via scikit-learn)",
        min_coverage=0.60,
        required_inputs=(),
        notes="30 nucleus measurements; grouped into mean / error / worst.",
    ),
    "stroke": TaskSpec(
        task_id="stroke",
        label="Stroke risk (Kaggle)",
        short_label="Stroke risk",
        kind="binary",
        tier=2,
        dataset="Kaggle Stroke Prediction Dataset",
        min_coverage=0.60,
        required_inputs=("age", "sex"),
    ),
    "thyroid": TaskSpec(
        task_id="thyroid",
        label="Hypothyroidism (UCI Thyroid)",
        short_label="Hypothyroidism",
        kind="binary",
        tier=2,
        dataset="UCI Thyroid Disease - hypothyroid.data",
        min_coverage=0.45,
        required_inputs=("age", "sex"),
    ),
}

TIER1_TASK = "symptoms"
TIER2_TASKS = [t for t, s in TASKS.items() if s.tier == 2]
ALL_TASKS = list(TASKS)

APP_VERSION = "2.0.0"


# --------------------------------------------------------------------------- #
# Secrets (.env) - never hard-code credentials
# --------------------------------------------------------------------------- #
def env(name: str, default: str = "") -> str:
    """Read an environment variable (values come from .env via python-dotenv)."""
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except Exception:  # dotenv is optional at runtime
        pass
    return os.environ.get(name, default)
