import numpy as np
import pandas as pd
import pytest
import yaml

from src.features import build_pipeline, clean_frame


@pytest.mark.parametrize("bad", ["Scr_IP", "Des_port", "class1", "class2", "class3"])
def test_forbidden_columns_never_reach_the_model(fitted_pipe, bad):
    assert bad not in fitted_pipe.named_steps["clean"].columns_


def test_default_hyperparameters_are_the_notebook_ones():
    clf = build_pipeline().named_steps["clf"]
    assert clf.C == 100
    assert clf.class_weight is None
    assert clf.max_iter == 500
    assert clf.tol == 1e-3


def test_params_yaml_matches_notebook():
    p = yaml.safe_load(open("configs/params.yaml", encoding="utf-8"))
    assert p["model"]["C_grid"] == [0.01, 0.1, 1, 10, 100]
    assert p["model"]["max_iter"] == 500
    assert p["model"]["tol"] == 0.001
    assert p["split"]["test_size"] == 0.2
    assert p["split"]["random_state"] == 42
    assert p["data"]["target"] == "class3"


def test_dash_becomes_missing_value():
    df = pd.DataFrame({"Duration": ["-", "2.5"], "Service": ["-", "http"]})
    out = clean_frame(df)
    assert out["Duration"].isna().tolist() == [True, False]
    assert out["Service"].isna().tolist() == [True, False]


def test_true_false_become_one_zero():
    out = clean_frame(pd.DataFrame({"anomaly_alert": ["TRUE", "FALSE", True, False]}))
    assert out["anomaly_alert"].tolist() == [1, 0, 1, 0]


def test_clipping_limits_extreme_values(fitted_pipe):
    prep = fitted_pipe.named_steps["prep"]
    clean = fitted_pipe.named_steps["clean"]
    normal = pd.DataFrame([{"Protocol": "tcp", "Service": "http", "Duration": 1.0,
                            "Scr_bytes": 10.0, "anomaly_alert": 0}])
    a = prep.transform(clean.transform(normal.assign(Scr_bytes=1e12)))
    b = prep.transform(clean.transform(normal.assign(Scr_bytes=1e15)))
    assert np.isfinite(a).all()
    assert np.allclose(a, b)   # au-delà du 99e centile, tout est ramené à la même valeur


def test_partial_row_is_accepted(fitted_pipe):
    proba = fitted_pipe.predict_proba(pd.DataFrame([{"Duration": 3.0}]))[0, 1]
    assert 0.0 <= proba <= 1.0


def test_unknown_category_is_ignored(fitted_pipe):
    row = pd.DataFrame([{"Protocol": "inconnu", "Duration": 3.0}])
    assert 0.0 <= fitted_pipe.predict_proba(row)[0, 1] <= 1.0


def test_completely_empty_row_is_accepted(fitted_pipe):
    row = pd.DataFrame([{"Duration": None, "Protocol": None}])
    assert 0.0 <= fitted_pipe.predict_proba(row)[0, 1] <= 1.0