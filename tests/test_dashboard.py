import pytest
import yaml
from conftest import make_df
from fastapi.testclient import TestClient

import src.api.main as api

client = TestClient(api.app)


@pytest.fixture
def with_model(monkeypatch, tmp_path, fitted_pipe):
    monkeypatch.setattr(api, "_model", fitted_pipe)
    monkeypatch.setattr(api, "LOG_PATH", tmp_path / "requests.csv")
    monkeypatch.setattr(api, "FEATURES_PATH", tmp_path / "absent_features.json")
    monkeypatch.setattr(api, "DRIFT_SUMMARY", tmp_path / "absent_drift.json")
    return tmp_path


def test_dashboard_page_is_served():
    r = client.get("/dashboard")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "IDS Console" in r.text


def test_root_redirects_to_dashboard():
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/dashboard"


def test_model_info_lists_columns(with_model):
    body = client.get("/model-info").json()
    assert body["model_loaded"] is True
    assert "Duration" in body["columns"]
    assert "class3" not in body["columns"]


def test_explain_returns_top_contributions(with_model):
    r = client.post("/explain", json={"data": {"Duration": 20.0, "Scr_bytes": 100}})
    assert r.status_code == 200
    items = r.json()["contributions"]
    assert 1 <= len(items) <= 8
    assert {"feature", "contribution", "direction"} <= set(items[0])


def test_explain_rejects_bad_data(with_model):
    assert client.post("/explain", json={"data": {"Duration": "abc"}}).status_code == 422


def test_stats_empty_when_no_log(with_model):
    s = client.get("/stats").json()
    assert s["total"] == 0 and s["recent"] == [] and s["attack_rate"] is None


def test_stats_counts_requests(with_model):
    client.post("/predict", json={"data": {"Duration": 1.0}})
    client.post("/predict", json={"data": {"Duration": 30.0}})
    s = client.get("/stats").json()
    assert s["total"] == 2
    assert s["attacks"] + s["normal"] == 2
    assert sum(s["histogram"]) == 2


def _params_with_val(tmp_path, csv_path):
    p = yaml.safe_load(open("configs/params.yaml", encoding="utf-8"))
    p["data"]["val_csv"] = str(csv_path)
    f = tmp_path / "params.yaml"
    f.write_text(yaml.safe_dump(p), encoding="utf-8")
    return f


def test_sample_returns_404_when_validation_file_missing(monkeypatch, with_model):
    monkeypatch.setattr(api, "PARAMS_PATH", _params_with_val(with_model, with_model / "absent.csv"))
    assert client.get("/sample?kind=attack").status_code == 404


def test_sample_returns_real_flow(monkeypatch, with_model):
    csv = with_model / "val.csv"
    make_df(300, 1)[0].to_csv(csv, index=False)
    monkeypatch.setattr(api, "PARAMS_PATH", _params_with_val(with_model, csv))
    r = client.get("/sample?kind=attack")
    assert r.status_code == 200
    assert r.json()["true_label"] == "Attack"
    assert "class3" not in r.json()["data"]
    assert client.post("/predict", json={"data": r.json()["data"]}).status_code == 200