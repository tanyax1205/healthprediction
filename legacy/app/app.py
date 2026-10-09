"""MultiDx - minimal local Streamlit app.

Run from the project root:

    .venv/bin/streamlit run app/app.py

The first launch fits one fast baseline per task and caches it under
``models/`` so subsequent launches are instant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Make ``src`` importable no matter where streamlit is launched from.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    APP_VERSION,
    DISCLAIMER,
    TASKS,
    TIER2_TASKS,
)
from src.train import ensure_models, load_bundle  # noqa: E402

st.set_page_config(page_title="MultiDx", page_icon="🩺", layout="wide")


# --------------------------------------------------------------------------- #
# Cached model loading / training
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def _get_bundles():
    """Load cached baselines, training any that are missing (once per process)."""
    return ensure_models()


def _format_value(name: str, raw) -> str:
    if isinstance(raw, float) and raw.is_integer():
        return f"{int(raw)}"
    return str(raw)


# --------------------------------------------------------------------------- #
# Sidebar: task picker
# --------------------------------------------------------------------------- #
st.sidebar.title("🩺 MultiDx")
st.sidebar.caption(f"v{APP_VERSION} · preliminary research demo")

task_options = {
    tid: TASKS[tid].label for tid in TASKS
}
selected = st.sidebar.radio(
    "Prediction task",
    options=list(task_options),
    format_func=lambda t: task_options[t],
)

with st.sidebar.expander("About"):
    st.markdown(
        "Baseline models trained on the bundled processed datasets. "
        "Tier 1 = symptom differential, Tier 2 = clinical risk tasks. "
        f"_{DISCLAIMER}_"
    )

st.sidebar.divider()
if st.sidebar.button("Retrain all baselines"):
    st.cache_resource.clear()
    ensure_models(force=True)
    st.sidebar.success("Models retrained.")
    st.rerun()


# --------------------------------------------------------------------------- #
# Load models
# --------------------------------------------------------------------------- #
with st.spinner("Preparing models (first launch trains quick baselines)…"):
    bundles = _get_bundles()

bundle = bundles.get(selected) or load_bundle(selected)

st.title(task_options[selected])
spec = TASKS[selected]
st.caption(f"{spec.dataset} · {spec.kind} · tier {spec.tier}")


# --------------------------------------------------------------------------- #
# Metrics strip
# --------------------------------------------------------------------------- #
if bundle:
    m = bundle["metrics"]
    cols = st.columns(4)
    cols[0].metric("Accuracy", f"{m['accuracy']:.1%}")
    if "roc_auc" in m:
        cols[1].metric("ROC AUC", f"{m['roc_auc']:.3f}")
        cols[2].metric("F1", f"{m['f1']:.3f}")
    else:
        cols[1].metric("F1 (macro)", f"{m['f1_macro']:.3f}")
        cols[2].metric("Classes", m.get("n_classes", "—"))
    cols[3].metric("Features", m["n_features"])

st.divider()


# --------------------------------------------------------------------------- #
# Input form
# --------------------------------------------------------------------------- #
def _symptom_input() -> pd.DataFrame:
    """Multi-select for binary flag tasks (e.g. 41-disease symptom checker)."""
    labels = bundle["features"]
    chosen = st.multiselect(
        "Select all symptoms that apply",
        options=labels,
        help="Leave empty to see the model's prior distribution.",
    )
    row = {f: (1 if f in chosen else 0) for f in labels}
    return pd.DataFrame([row])


def _generic_input() -> pd.DataFrame:
    meta = bundle["feature_meta"]
    row: dict = {}
    cats = bundle["categorical"]
    nums = bundle["numeric"]

    if nums:
        st.subheader("Numeric inputs")
        ncols = min(3, len(nums))
        grid = st.columns(ncols)
        for i, col in enumerate(nums):
            info = meta[col]
            default = float(info["median"])
            lo = float(info["min"])
            hi = float(info["max"])
            # Guard against degenerate ranges so number_input never errors.
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = default - 1.0, default + 1.0
            with grid[i % ncols]:
                row[col] = st.number_input(
                    col, value=default, min_value=lo, max_value=hi, key=f"num_{col}"
                )

    if cats:
        st.subheader("Categorical inputs")
        grid = st.columns(min(3, len(cats)))
        for i, col in enumerate(cats):
            options = meta[col]["options"] or ["unknown"]
            with grid[i % len(cats)]:
                row[col] = st.selectbox(col, options=options, key=f"cat_{col}")

    return pd.DataFrame([row])


with st.form("predict_form"):
    if bundle and bundle.get("multiselect"):
        X_input = _symptom_input()
    elif bundle:
        X_input = _generic_input()
    else:
        st.error("No model available for this task.")
        X_input = None
    submitted = st.form_submit_button("Predict", use_container_width=True)


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
if submitted and bundle and X_input is not None:
    model = bundle["pipeline"]
    if bundle["kind"] == "binary":
        proba = float(model.predict_proba(X_input)[0, 1])
        st.subheader("Result")
        st.metric("Risk probability", f"{proba:.1%}")
        band = "High" if proba >= 0.66 else "Moderate" if proba >= 0.33 else "Low"
        st.progress(min(max(proba, 0.0), 1.0))
        st.write(f"**Risk band:** {band}")
    else:
        proba = model.predict_proba(X_input)[0]
        classes = bundle["classes"]
        order = np.argsort(proba)[::-1][:5]
        st.subheader("Top differential")
        for rank, idx in enumerate(order, start=1):
            st.write(f"{rank}. **{classes[idx]}** — {proba[idx]:.1%}")
            st.progress(min(max(float(proba[idx]), 0.0), 1.0))

    st.caption(f"⚠️ {DISCLAIMER}")
