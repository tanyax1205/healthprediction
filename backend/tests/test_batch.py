"""Batch CSV prediction endpoint."""

FULL_HEADER = (
    "age,sex,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,cp\n"
)
ROW1 = "63,1,145,233,1,0,150,0,2.3,1,0,3,3\n"
ROW2 = "55,0,130,210,1,1,160,0,1.5,2,0,7,4\n"
SPARSE_HEADER = "age,sex,trestbps,chol\n"


def _upload(client, content: str, name: str = "rows.csv"):
    return client.post(
        "/predict/heart/batch",
        files={"file": (name, content, "text/csv")},
    )


def test_batch_valid_rows(client):
    resp = _upload(client, FULL_HEADER + ROW1 + ROW2)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    text = resp.text
    lines = text.strip().splitlines()
    assert len(lines) == 3  # header + 2 rows
    assert "predicted_label" in lines[0]
    assert "risk_level" in lines[0]


def test_batch_per_row_validation_errors(client):
    resp = _upload(client, SPARSE_HEADER + "63,1,145,233\n")
    assert resp.status_code == 200
    lines = resp.text.strip().splitlines()
    assert "warnings" in lines[0]


def test_batch_missing_required_column_is_400(client):
    resp = _upload(client, "chol\n233\n")
    assert resp.status_code == 400


def test_batch_empty_file_is_400(client):
    resp = _upload(client, "")
    assert resp.status_code == 400


def test_batch_garbage_is_400(client):
    resp = _upload(client, "this is not csv \n \n")
    assert resp.status_code == 400