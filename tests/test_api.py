import pytest
from fastapi.testclient import TestClient

import src.api.main as api

client = TestClient(api.app)


@pytest.fixture
def with_model(monkeypatch, tmp_path, fitted_pipe):
    monkeypatch.setattr(api, "_model", fitted_pipe)
    monkeypatch.setattr(api, "LOG_PATH", tmp_path / "requests.csv")
    monkeypatch.setattr(api, "FEATURES_PATH", tmp_path / "absent_features.json")
    return tmp_path / "requests.csv"


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_body_missing_returns_422_by_default():
    assert client.post("/predict").status_code == 422


def test_predict_wrong_structure_returns_422():
    assert client.post("/predict", json={"foo": 1}).status_code == 422
    assert client.post("/predict", json={"data": "abc"}).status_code == 422


def test_predict_empty_data_returns_422():
    assert client.post("/predict", json={"data": {}}).status_code == 422


def test_validation_status_400_when_configured(monkeypatch):
    monkeypatch.setenv("VALIDATION_STATUS", "400")
    assert client.post("/predict").status_code == 400


def test_unknown_column_returns_400_when_configured(monkeypatch, with_model):
    monkeypatch.setenv("VALIDATION_STATUS", "400")
    assert client.post("/predict", json={"data": {"colonne_bidon": 1}}).status_code == 400


def test_predict_ok(with_model):
    r = client.post("/predict", json={"data": {"Protocol": "tcp", "Duration": 2.0,
                                               "Scr_bytes": 100, "anomaly_alert": "FALSE"}})
    assert r.status_code == 200
    assert r.json()["prediction"] in ("Normal", "Attack")
    assert 0 <= r.json()["probability_attack"] <= 1


def test_predict_partial_row_ok(with_model):
    assert client.post("/predict", json={"data": {"Duration": 2.0}}).status_code == 200


def test_unknown_column_returns_422(with_model):
    assert client.post("/predict", json={"data": {"colonne_bidon": 1}}).status_code == 422


def test_text_in_numeric_column_returns_422(with_model):
    assert client.post("/predict", json={"data": {"Duration": "abc"}}).status_code == 422


def test_request_is_logged(with_model):
    client.post("/predict", json={"data": {"Duration": 2.0}})
    assert with_model.exists()
    assert "prediction" in with_model.read_text()


def test_missing_model_returns_503(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "_model", None)
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "absent.pkl")
    assert client.post("/predict", json={"data": {"Duration": 1.0}}).status_code == 503