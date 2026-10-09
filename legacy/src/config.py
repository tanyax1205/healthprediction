"""Central configuration for MultiDx: paths, seeds, task registry and model grids.

Paper reference: Table II (tuning grids) and Section III (experimental setup).
All randomness in the project is controlled by SEED.
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
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
MODEL_CARDS_DIR = REPORTS_DIR / "model_cards"
TRIPOD_DIR = REPORTS_DIR / "tripod"
RESULTS_DIR = ROOT / "results"
APP_DIR = ROOT / "app"

for _d in (
    DATA_RAW,
    DATA_PROCESSED,
    MODELS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_CARDS_DIR,
    TRIPOD_DIR,
    RESULTS_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
TEST_SIZE = 0.20
VAL_FRACTION = 0.15  # carved from the training portion, used ONLY for risk cut-offs
CV_FOLDS = 5
BOOTSTRAP_N = 2000

DISCLAIMER = "Preliminary model-based indication. Not a medical diagnosis."

# --------------------------------------------------------------------------- #
# Six classifiers compared everywhere (paper Section IV)
# --------------------------------------------------------------------------- #
MODEL_NAMES = ["logreg", "decision_tree", "random_forest", "svm", "knn", "xgboost"]

MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm": "SVM",
    "knn": "KNN",
    "xgboost": "XGBoost",
}

# Trees do not need scaling; LR/SVM/KNN do (paper Section III-B).
NEEDS_SCALING = {
    "logreg": True,
    "decision_tree": False,
    "random_forest": False,
    "svm": True,
    "knn": True,
    "xgboost": False,
}

# How class imbalance is handled per model (training folds only).
IMBALANCE_STRATEGY = {
    "logreg": "class_weight",
    "decision_tree": "class_weight",
    "random_forest": "class_weight",
    "svm": "class_weight",
    "knn": "smote",
    "xgboost": "scale_pos_weight",
}

# Preference used when CV scores are statistically close (paper Section IV-C):
# models whose fitted estimators are directly supported by SHAP's fast
# Tree/Linear explainers are preferred over the KernelExplainer fallback.
SHAP_PREFERENCE = {
    "random_forest": 0,
    "xgboost": 0,
    "decision_tree": 0,
    "logreg": 0,
    "svm": 0,
    "knn": 1,
}

# --------------------------------------------------------------------------- #
# Tuning grids - paper Table II
# --------------------------------------------------------------------------- #
PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "logreg": {
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__penalty": ["l2"],
    },
    "decision_tree": {
        "clf__max_depth": [3, 5, 7, 10, None],
        "clf__min_samples_leaf": [1, 5, 10],
    },
    "random_forest": {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [None, 10, 20],
        "clf__max_features": ["sqrt", 0.5],
    },
    "svm": {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__kernel": ["rbf", "linear"],
        "clf__gamma": ["scale", 0.01],
    },
    "knn": {
        "clf__n_neighbors": [3, 5, 7, 11],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["minkowski", "manhattan"],
    },
    "xgboost": {
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [3, 5, 7],
        "clf__subsample": [0.8, 1.0],
    },
}

# --------------------------------------------------------------------------- #
# Risk / calibration configuration
# --------------------------------------------------------------------------- #
RISK_CONFIG: dict[str, Any] = {
    # Probability band (grey zone) around a cut-off: inside it we abstain.
    "grey_zone": 0.05,
    # Target precision used to pick the Moderate/High cut-off on validation data.
    "target_precision_high": 0.60,
    # Fallback quantiles if the precision rule cannot be met on validation data.
    "quantile_low": 0.33,
    "quantile_high": 0.66,
    # Minimum calibrated top-1 probability to issue any non-inconclusive result.
    "min_confidence": 0.45,
}

OOD_CONFIG: dict[str, Any] = {
    "mahalanobis_chi2": 0.99,  # reject if distance^2 above this quantile of train
    "range_slack": 0.10,  # relative slack for per-feature min/max check
}

# --------------------------------------------------------------------------- #
# Task registry - Tier 1 (symptoms, multi-class) + Tier 2 (clinical, binary)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TaskSpec:
    """One prediction task."""

    task_id: str
    label: str
    kind: str  # "multiclass" | "binary"
    tier: int
    dataset: str
    # Fraction of model input features that must be present in the unified
    # profile before the model is allowed to answer (spec requirement:
    # "mostly missing" inputs must NOT produce a guess).
    min_coverage: float
    # Features that must all be present regardless of coverage.
    required_inputs: tuple[str, ...] = field(default_factory=tuple)
    # Profile fields (see app_common.PROFILE_TO_FEATURE) consumed, for docs.
    notes: str = ""


TASKS: dict[str, TaskSpec] = {
    "symptoms": TaskSpec(
        task_id="symptoms",
        label="Symptom-based differential (41 diseases)",
        kind="multiclass",
        tier=1,
        dataset="Kaggle: Disease Symptom Prediction",
        min_coverage=0.0,
        notes="Multi-class; answered by the adaptive symptom checker page.",
    ),
    "heart": TaskSpec(
        task_id="heart",
        label="Heart disease (UCI Cleveland)",
        kind="binary",
        tier=2,
        dataset="UCI Heart Disease - processed.cleveland.data",
        min_coverage=0.50,
        required_inputs=("age", "sex", "trestbps"),
    ),
    "diabetes": TaskSpec(
        task_id="diabetes",
        label="Diabetes (Pima Indians)",
        kind="binary",
        tier=2,
        dataset="Pima Indians Diabetes Database (UCI/NIDDK mirror)",
        min_coverage=0.50,
        required_inputs=("age", "glucose"),
    ),
    "liver": TaskSpec(
        task_id="liver",
        label="Liver disease (ILPD)",
        kind="binary",
        tier=2,
        dataset="UCI Indian Liver Patient Dataset (Kaggle mirror)",
        min_coverage=0.50,
        required_inputs=("age", "sex"),
    ),
    "ckd": TaskSpec(
        task_id="ckd",
        label="Chronic kidney disease (UCI CKD)",
        kind="binary",
        tier=2,
        dataset="UCI Chronic Kidney Disease (arff)",
        min_coverage=0.50,
        required_inputs=("age", "bp"),
    ),
    "cancer": TaskSpec(
        task_id="cancer",
        label="Breast cancer malignancy (WDBC)",
        kind="binary",
        tier=2,
        dataset="UCI Wisconsin Diagnostic Breast Cancer (via scikit-learn)",
        min_coverage=0.60,
        required_inputs=(),
        notes="30 nucleus measurements; usually only available from a pathology report.",
    ),
    "stroke": TaskSpec(
        task_id="stroke",
        label="Stroke risk (Kaggle)",
        kind="binary",
        tier=2,
        dataset="Kaggle Stroke Prediction Dataset",
        min_coverage=0.60,
        required_inputs=("age", "sex"),
    ),
    "thyroid": TaskSpec(
        task_id="thyroid",
        label="Hypothyroidism (UCI Thyroid)",
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

# --------------------------------------------------------------------------- #
# Fairness / subgroup configuration
# --------------------------------------------------------------------------- #
AGE_BANDS: list[tuple[str, int, int]] = [
    ("under 40", 0, 39),
    ("40-59", 40, 59),
    ("60 and over", 60, 200),
]
FAIRNESS_GAP_FLAG = 0.10  # flag subgroup recall gaps >= 10 percentage points

# Column name that holds the subgroup sex/gender per task (None => unavailable)
FAIRNESS_SEX_COLUMN: dict[str, str | None] = {
    "heart": "sex",
    "diabetes": None,
    "liver": "gender",
    "ckd": None,
    "cancer": None,
    "stroke": "gender",
    "thyroid": "sex",
    "symptoms": None,
}

# --------------------------------------------------------------------------- #
# OCR / lab analytes
# --------------------------------------------------------------------------- #
SUPPORTED_LAB_UNITS: dict[str, list[str]] = {
    "glucose": ["mg/dl", "mmol/l"],
    "cholesterol": ["mg/dl", "mmol/l"],
    "blood_pressure": ["mmhg"],
    "bmi": ["kg/m2"],
    "creatinine": ["mg/dl", "umol/l"],
    "bilirubin": ["mg/dl", "umol/l"],
    "albumin": ["g/dl", "g/l"],
    "tsh": ["miu/l", "mu/l"],
}

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


APP_VERSION = "1.0.0"
