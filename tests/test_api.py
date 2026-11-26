"""Tests for app/main.py through FastAPI's TestClient.

Needs only models/model.joblib and params.yaml (both in git), never a CSV.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from mp093.config import load_params

client = TestClient(app)

VALID = {
    "cards_attached": 742,
    "working_hours": 10.5,
    "biometric_failures": 55,
    "temp_c": 38.5,
    "stock_kg_morning": 1497.0,
    "days_to_month_end": 5,
    "shop_area": "tribal",
    "block": "Block North",
}


def test_valid_body_has_exactly_the_params_features():
    # Keeps VALID in step with params.yaml (and so with PROBLEM.md section 5).
    assert list(VALID) == load_params()["features"]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_valid_request():
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    assert isinstance(r.json()["prediction"], float)


@pytest.mark.parametrize("field, value", [
    ("working_hours", 16),      # client rule 1: above 15 is impossible
    ("cards_attached", 99),     # client rule 2: below 100 is a typing mistake
])
def test_client_rules_give_422(field, value):
    r = client.post("/predict", json={**VALID, field: value})
    assert r.status_code == 422


def test_rule_edges_are_allowed():
    r = client.post("/predict", json={**VALID, "working_hours": 15, "cards_attached": 100})
    assert r.status_code == 200


def test_missing_field_gives_422():
    body = {k: v for k, v in VALID.items() if k != "block"}
    assert client.post("/predict", json=body).status_code == 422


def test_extra_field_gives_422():
    body = {**VALID, "customer_name": "someone"}
    assert client.post("/predict", json=body).status_code == 422


def test_one_prediction_writes_one_json_log_line(capsys):
    capsys.readouterr()                      # drop anything printed before
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200

    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    records = [json.loads(ln) for ln in lines]          # every line is valid JSON
    predictions = [rec for rec in records if rec.get("event") == "prediction"]
    assert len(predictions) == 1

    rec = predictions[0]
    for key in ("timestamp", "level", "logger", "message", "latency_ms", *VALID):
        assert key in rec
    assert rec["prediction"] == r.json()["prediction"]


def test_rejected_request_writes_no_prediction_log(capsys):
    capsys.readouterr()
    client.post("/predict", json={**VALID, "working_hours": 16})
    assert '"event": "prediction"' not in capsys.readouterr().out
