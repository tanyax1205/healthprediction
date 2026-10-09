# MultiDx (`healtprediction_mp`)

Multi-disease risk screening across 8 datasets (symptom differential + 7
clinical risk tasks). Research-grade ML pipeline + FastAPI backend (Step 2 ✅)
+ React frontend (in progress).

> ⚠️ Preliminary model-based indication. Not a medical diagnosis.

## Repo layout

```
ml/          training + evaluation + saved artifacts (Step 1 ✅)
backend/     FastAPI service (Step 2 ✅)
frontend/    React 18 + Vite + TS   (next)
data/        raw + processed datasets (immutable, committed)
legacy/      old Streamlit app + baseline trainer (reference only)
reports/     results.csv / results.tex + metrics
```

## ML pipeline (`ml/`)

```bash
cd healtprediction_mp
python -m venv .venv && .venv/bin/pip install -r ml/requirements.txt

# quick smoke run (logreg + rf, 3 candidates)
.venv/bin/python -m ml.train --all --quick

# full run: 5 models/task (LR, RF, XGBoost, LightGBM, SVM),
# stratified 5-fold CV + RandomizedSearchCV, calibration, thresholds (60s)
.venv/bin/python -m ml.train --all
.venv/bin/python -m ml.train --task heart --models logreg svm --n-iter 8

# export results table (CSV + LaTeX for the IEEE paper)
.venv/bin/python -m ml.export

# tests + lint
.venv/bin/python -m pytest
.venv/bin/ruff check ml
```

Artifacts are saved per task/version under `ml/artifacts/<task>/<version>/`
(model.joblib + metadata.json) with `ml/artifacts/index.json` pointing at the
latest version. Each artifact carries: version, training date, processed-data
SHA-256, model, params, CV results, metrics (accuracy/precision/recall/F1
/ROC-AUC/PR-AUC/Brier), confusion matrix, ROC + calibration curves, threshold,
and an `shap_available` flag.

Design notes
- Preprocessing (imputer + scaler + encoder) is inside each sklearn Pipeline,
  fitted on training folds only — no data leakage.
- Class imbalance handled by `class_weight`/`scale_pos_weight` (fold-safe).
- Binary operating thresholds are chosen on **out-of-fold calibrated**
  training probabilities to prioritise recall for the disease-positive class
  (min recall 0.60), never on the test set.
- The 41-class symptom task is near-deterministic (100% accuracy is expected):
  reported with a caveat, never advertised as performance.
- macOS note: LightGBM needs `brew install libomp` once.

## Backend API (`backend/`)

```bash
cd healtprediction_mp
.venv/bin/pip install -r backend/requirements.txt
make api            # or: .venv/bin/python -m backend.run --reload
```

Interactive docs: <http://localhost:8000/docs>. The runner binds to
`API_HOST`/`API_PORT` from `.env` (default `0.0.0.0:8000`, i.e. also reachable
from other devices on your LAN). Models are loaded once at startup from
`MULTIDX_ARTIFACTS` (default `ml/artifacts`); run
`.venv/bin/python -m ml.train --all` first if the index is missing.

Quick sanity check against a running server: `make smoke`.

| Endpoint | Purpose |
| --- | --- |
| `GET /tasks` | all tasks + dynamic feature schemas (name, label, type, unit, min/max, default, group) |
| `GET /tasks/{task}` | schema for one task |
| `POST /predict/{task}` | single prediction (calibrated probability, risk level, top-3/5 SHAP, outlier warnings, model version) |
| `POST /predict/{task}/batch` | CSV upload → annotated CSV (per-row validation errors) |
| `GET /tasks/{task}/metrics` | held-out metrics, confusion matrix, ROC + calibration curves, CV results |
| `GET /health` · `GET /version` | liveness + model versions |

Environment: `API_RATE_LIMIT` (per-IP/minute, `0` disables), `API_CORS_ORIGINS`,
`API_LOG_LEVEL`. Retraining is only exposed via the CLI (`python -m ml.train`),
never through the API.

```bash
# backend tests + lint
.venv/bin/python -m pytest backend/tests
.venv/bin/ruff check backend ml
```

## Env / deployment

See `.env.example`. Frontend + Docker/CI are in progress.