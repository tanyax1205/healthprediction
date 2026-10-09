"""Tests for the feature/task schema."""

from __future__ import annotations

import json

from ml.config import TASKS
from ml.data_loader import load_task
from ml.schema.features import active_features, build_feature_specs, task_schema


def test_schema_covers_all_tasks():
    for task in TASKS:
        schema = task_schema(task)
        assert schema["task_id"] == task
        assert len(schema["features"]) > 0
        # every field the frontend needs
        for f in schema["features"]:
            assert set(f) >= {
                "name", "label", "type", "unit", "group", "min", "max", "default", "options", "required",
            }
            assert f["type"] in {"numeric", "categorical", "binary"}


def test_heart_required_and_no_constants():
    df, _ = load_task("heart")
    features = active_features(df)
    assert "age" in features and "bpm" not in features
    specs = {s.name: s for s in build_feature_specs("heart", df)}
    assert specs["age"].required is True
    assert specs["sex"].required is True
    assert specs["trestbps"].unit == "mm Hg"


def test_symptoms_schema_is_binary_and_large():
    schema = task_schema("symptoms")
    assert schema["kind"] == "multiclass"
    assert len(schema["features"]) > 100
    assert all(f["type"] == "binary" for f in schema["features"])


def test_schema_json_serialisable():
    json.dumps(task_schema("cancer"))
