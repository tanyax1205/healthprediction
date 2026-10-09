"""Single prediction endpoint."""

HEART_VALUES = {
    "age": 63, "sex": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3,
    "slope": 1, "ca": 0, "thal": 3, "cp": 3,
}


def test_predict_valid(client):
    resp = client.post("/predict/heart", json={"values": HEART_VALUES})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in {"Positive", "Negative"}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_level"] in {"Low", "Moderate", "High"}
    assert body["model_version"]
    assert body["disclaimer"]
    assert len(body["top_classes"]) >= 2


def test_predict_shap_for_explainable_model(client):
    body = client.post("/predict/heart", json={"values": HEART_VALUES}).json()
    assert len(body["shap"]) == 5
    assert all(c["feature"] for c in body["shap"])


def test_predict_missing_required_is_422(client):
    resp = client.post("/predict/heart", json={"values": {"chol": 233}})
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert any(d["feature"] == "age" for d in detail)


def test_predict_out_of_range_warns(client):
    values = dict(HEART_VALUES, thalach=230)  # max trained is ~202
    body = client.post("/predict/heart", json={"values": values}).json()
    assert body["outlier_warnings"]
    assert body["outlier_warnings"][0]["feature"] == "thalach"


def test_predict_unknown_task_404(client):
    resp = client.post("/predict/nope", json={"values": HEART_VALUES})
    assert resp.status_code == 404


def test_predict_bad_value_type_is_422(client):
    values = dict(HEART_VALUES, age="old")
    resp = client.post("/predict/heart", json={"values": values})
    assert resp.status_code == 422


def test_metrics_endpoint(client):
    resp = client.get("/tasks/heart/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["best_model"]
    assert body["metrics"]["accuracy"] > 0
    assert len(body["confusion_matrix"]) == 2
    assert "roc" in body["curves"]
    assert body["cv_results"]