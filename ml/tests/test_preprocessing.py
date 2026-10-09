"""Tests for leakage-safe preprocessing pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ml.config import NEEDS_SCALING
from ml.preprocessing import build_pipeline, split_feature_types


def _toy_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame(
        {
            "age": rng.integers(20, 80, n),
            "chol": rng.normal(200, 30, n),
            "sex": rng.choice(["M", "F"], n),
            "target": rng.integers(0, 2, n),
        }
    )


def test_pipeline_is_sklearn_pipeline_with_pre_and_clf():
    df = _toy_df()
    numeric, categorical = split_feature_types(df, ["age", "chol", "sex"])
    pipe = build_pipeline("logreg", "binary", numeric, categorical, seed=42)
    assert isinstance(pipe, Pipeline)
    assert "pre" in pipe.named_steps and "clf" in pipe.named_steps
    pipe.fit(df[["age", "chol", "sex"]], df["target"])
    # scaling lives on the numeric branch only
    num_branch = pipe.named_steps["pre"].named_transformers_["num"]
    assert ("scale" in dict(num_branch.steps)) == NEEDS_SCALING["logreg"]


def test_fit_predict_shape_and_no_nan():
    df = _toy_df()
    numeric, categorical = split_feature_types(df, ["age", "chol", "sex"])
    pipe = build_pipeline("random_forest", "binary", numeric, categorical, seed=42)
    pipe.fit(df[["age", "chol", "sex"]], df["target"])
    proba = pipe.predict_proba(df[["age", "chol", "sex"]])
    assert proba.shape == (len(df), 2)
    assert np.isfinite(proba).all()


def test_categorical_unseen_level_does_not_crash():
    df = _toy_df()
    numeric, categorical = split_feature_types(df, ["age", "chol", "sex"])
    pipe = build_pipeline("logreg", "binary", numeric, categorical, seed=42)
    pipe.fit(df[["age", "chol", "sex"]], df["target"])
    unseen = pd.DataFrame({"age": [40], "chol": [180.0], "sex": ["X"]})
    assert pipe.predict_proba(unseen).shape == (1, 2)
