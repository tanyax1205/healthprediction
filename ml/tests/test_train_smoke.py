"""Smoke test: end-to-end training + explanation on a tiny task/mode."""

from __future__ import annotations

import json

from ml.config import ARTIFACTS_DIR
from ml.data_loader import load_task
from ml.explain import explain_row
from ml.schema.features import active_features
from ml.train import train_task


def test_train_task_smoke_heart():
    meta = train_task("heart", models=["logreg", "random_forest"], n_iter=2)
    assert meta["task"] == "heart"
    assert meta["best_model"] in {"logreg", "random_forest"}
    assert 0.0 <= meta["metrics"]["roc_auc"] <= 1.0
    assert meta["dataset_hash"]
    assert (ARTIFACTS_DIR / "heart" / meta["version"] / "model.joblib").exists()


def test_explain_row_returns_at_most_topk():
    import joblib

    df, _ = load_task("heart")
    features = active_features(df)
    idx_path = ARTIFACTS_DIR / "heart"
    # use the artifact written by the previous test
    version = json.loads((ARTIFACTS_DIR / "index.json").read_text())["heart"]
    bundle = joblib.load(idx_path / version / "model.joblib")

    row = df[features].iloc[[0]]
    contribs = explain_row(bundle["explain_model"], row, top_k=5)
    assert len(contribs) <= 5
    for c in contribs:
        assert set(c) >= {"feature", "contribution", "direction"}
