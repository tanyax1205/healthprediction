"""Feature metadata: labels, units, groups and value ranges.

The frontend renders form fields dynamically from this schema, so it must be
complete and self-describing (name, label, type, unit, group, min, max,
default, required).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from ml.config import DISCLAIMER, TASKS
from ml.data_loader import load_task

STANDARD_TARGET = "target"

# --------------------------------------------------------------------------- #
# Presentation metadata (hand-curated; the data files are never modified)
# --------------------------------------------------------------------------- #
LABEL_OVERRIDES: dict[str, str] = {
    "bp": "Blood pressure",
    "bgr": "Blood glucose (random)",
    "bu": "Blood urea",
    "sc": "Serum creatinine",
    "sg": "Urine specific gravity",
    "al": "Albumin",
    "su": "Sugar",
    "sod": "Sodium",
    "pot": "Potassium",
    "pcv": "Packed cell volume",
    "htn": "Hypertension",
    "dm": "Diabetes mellitus",
    "ane": "Anemia",
    "appet": "Appetite",
    "trestbps": "Resting blood pressure",
    "chol": "Serum cholesterol",
    "fbs": "Fasting blood sugar > 120 mg/dL",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate achieved",
    "exang": "Exercise induced angina",
    "oldpeak": "ST depression (oldpeak)",
    "ca": "Major vessels",
    "thal": "Thalassemia",
    "cp": "Chest pain type",
    "bmi": "Body mass index",
    "tsh": "TSH",
    "t3": "T3",
    "tt4": "TT4",
    "t4u": "T4U",
    "fti": "FTI",
    "avg_glucose_level": "Average glucose level",
    "alkaline_phosphatase": "Alkaline phosphatase",
    "alt": "ALT (SGPT)",
    "ast": "AST (SGOT)",
    "albumin_globulin_ratio": "Albumin/globulin ratio",
    "diabetes_pedigree": "Diabetes pedigree function",
    "ever_married": "Ever married",
    "work_type": "Work type",
    "residence_type": "Residence type",
    "smoking_status": "Smoking status",
}

UNITS: dict[str, str] = {
    "age": "years",
    "pregnancies": "count",
    "glucose": "mg/dL",
    "blood_pressure": "mm Hg",
    "skin_thickness": "mm",
    "insulin": "µU/mL",
    "bmi": "kg/m²",
    "trestbps": "mm Hg",
    "chol": "mg/dL",
    "thalach": "bpm",
    "ca": "vessels",
    "bp": "mm Hg",
    "sg": "g/mL",
    "bgr": "mg/dL",
    "bu": "mg/dL",
    "sc": "mg/dL",
    "sod": "mEq/L",
    "pot": "mEq/L",
    "hemo": "g/dL",
    "pcv": "%",
    "avg_glucose_level": "mg/dL",
    "total_bilirubin": "mg/dL",
    "direct_bilirubin": "mg/dL",
    "alkaline_phosphatase": "IU/L",
    "alt": "IU/L",
    "ast": "IU/L",
    "total_proteins": "g/dL",
    "albumin": "g/dL",
    "tsh": "mIU/L",
    "t3": "µg/dL",
    "tt4": "µg/dL",
    "t4u": "ratio",
}

# Explicit groups per task; anything unlisted falls back to a default group.
GROUPS: dict[str, dict[str, str]] = {
    "heart": {
        "age": "Demographics", "sex": "Demographics",
        "trestbps": "Vitals", "thalach": "Vitals",
        "chol": "Labs", "fbs": "Labs",
        "restecg": "ECG", "oldpeak": "ECG", "slope": "ECG", "ca": "ECG", "thal": "ECG",
        "cp": "Symptoms", "exang": "Symptoms",
    },
    "diabetes": {
        "age": "Demographics", "pregnancies": "Demographics",
        "diabetes_pedigree": "History",
        "blood_pressure": "Vitals", "bmi": "Vitals",
        "glucose": "Labs", "insulin": "Labs", "skin_thickness": "Labs",
    },
    "liver": {
        "age": "Demographics", "sex": "Demographics",
        "alkaline_phosphatase": "Liver enzymes", "alt": "Liver enzymes", "ast": "Liver enzymes",
        "total_bilirubin": "Bilirubin", "direct_bilirubin": "Bilirubin",
        "total_proteins": "Proteins", "albumin": "Proteins",
        "albumin_globulin_ratio": "Proteins",
    },
    "ckd": {
        "age": "Demographics",
        "bp": "Vitals",
        "sg": "Urine", "al": "Urine", "su": "Urine",
        "bgr": "Blood", "bu": "Blood", "sc": "Blood", "sod": "Blood", "pot": "Blood",
        "hemo": "Blood", "pcv": "Blood",
        "htn": "History", "dm": "History", "appet": "History", "ane": "History",
    },
    "stroke": {
        "gender": "Demographics", "age": "Demographics", "ever_married": "Demographics",
        "hypertension": "History", "heart_disease": "History", "smoking_status": "History",
        "work_type": "Lifestyle", "residence_type": "Lifestyle",
        "avg_glucose_level": "Labs", "bmi": "Labs",
    },
    "thyroid": {
        "age": "Demographics", "sex": "Demographics",
        "on_thyroxine": "History", "query_on_thyroxine": "History",
        "on_antithyroid_medication": "History", "thyroid_surgery": "History",
        "query_hypothyroid": "History", "query_hyperthyroid": "History",
        "pregnant": "History", "sick": "History", "tumor": "History",
        "lithium": "History", "goitre": "History",
        "tsh": "Labs", "t3": "Labs", "tt4": "Labs", "t4u": "Labs", "fti": "Labs",
    },
}

CANCER_GROUPS = {"mean": "Mean", "error": "Error", "worst": "Worst"}


def _label(name: str) -> str:
    if name in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[name]
    if name.startswith(("mean_", "worst_")) or name.endswith("_error"):
        return name.replace("_", " ").title()
    return name.replace("_", " ").capitalize()


def _group(task: str, name: str) -> str:
    if task == "symptoms":
        return "Symptoms"
    if task == "cancer":
        if name.startswith("mean_"):
            return CANCER_GROUPS["mean"]
        if name.endswith("_error"):
            return CANCER_GROUPS["error"]
        if name.startswith("worst_"):
            return CANCER_GROUPS["worst"]
    return GROUPS.get(task, {}).get(name, "Other")


def _col_type(series: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(series):
        return "categorical"
    values = series.dropna()
    if len(values) and values.isin([0, 1]).all() and values.nunique() == 2:
        return "binary"
    return "numeric"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    label: str
    type: str  # "numeric" | "categorical" | "binary"
    unit: str
    group: str
    min: float | None
    max: float | None
    default: float | str
    options: list[str]
    required: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def active_features(df: pd.DataFrame) -> list[str]:
    """Feature columns that carry signal (drop constants / all-missing)."""
    cols = [c for c in df.columns if c != STANDARD_TARGET]
    return [c for c in cols if df[c].nunique(dropna=True) > 1]


def build_feature_specs(task: str, df: pd.DataFrame) -> list[FeatureSpec]:
    spec = TASKS[task]
    required = set(spec.required_inputs)
    specs: list[FeatureSpec] = []
    for name in active_features(df):
        s = df[name]
        ftype = _col_type(s)
        if ftype == "categorical":
            options = sorted(s.dropna().astype(str).unique().tolist())
            default: float | str = options[0] if options else ""
            vmin = vmax = None
        else:
            vals = pd.to_numeric(s, errors="coerce").dropna()
            vmin = float(vals.min()) if len(vals) else 0.0
            vmax = float(vals.max()) if len(vals) else 1.0
            default = float(vals.median()) if len(vals) else 0.0
            options = []
        specs.append(
            FeatureSpec(
                name=name,
                label=_label(name),
                type=ftype,
                unit=UNITS.get(name, ""),
                group=_group(task, name),
                min=vmin,
                max=vmax,
                default=default,
                options=options,
                required=name in required,
            )
        )
    return specs


def task_schema(task: str) -> dict[str, Any]:
    """Full schema for one task, ready to serialise to the API."""
    if task not in TASKS:
        raise KeyError(f"unknown task '{task}'")
    df, _ = load_task(task)
    specs = build_feature_specs(task, df)
    spec = TASKS[task]
    groups: list[str] = []
    for s in specs:
        if s.group not in groups:
            groups.append(s.group)
    return {
        "task_id": spec.task_id,
        "label": spec.label,
        "short_label": spec.short_label,
        "kind": spec.kind,
        "tier": spec.tier,
        "dataset": spec.dataset,
        "min_coverage": spec.min_coverage,
        "disclaimer": DISCLAIMER,
        "groups": groups,
        "features": [s.to_dict() for s in specs],
    }


def all_task_schemas() -> dict[str, dict[str, Any]]:
    return {task: task_schema(task) for task in TASKS}
