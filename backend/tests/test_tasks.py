"""Task catalog + feature schema endpoint."""


def test_list_tasks_has_all_schemas(client):
    resp = client.get("/tasks")
    assert resp.status_code == 200
    tasks = resp.json()["tasks"]
    assert "heart" in tasks
    assert "symptoms" in tasks


def test_feature_schema_shape(client):
    heart = client.get("/tasks/heart").json()
    assert heart["kind"] == "binary"
    fields = {"name", "label", "type", "unit", "min", "max", "default", "group"}
    for feat in heart["features"]:
        assert fields <= set(feat)


def test_required_features_marked(client):
    heart = client.get("/tasks/heart").json()
    by_name = {f["name"]: f for f in heart["features"]}
    assert by_name["age"]["required"] is True
    assert by_name["trestbps"]["required"] is True


def test_unknown_task_404(client):
    assert client.get("/tasks/nope").status_code == 404