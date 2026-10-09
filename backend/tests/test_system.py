"""Health / version endpoints."""


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["tasks_loaded"] >= 1
    assert body["models_loaded"] >= 1


def test_version_lists_models(client):
    resp = client.get("/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"]
    assert "heart" in body["models"]


def test_root_points_to_docs(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["docs"] == "/docs"