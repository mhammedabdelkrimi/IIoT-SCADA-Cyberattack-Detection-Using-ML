import json

import joblib

from src.promote import promote, should_promote


def test_first_model_is_promoted():
    assert should_promote(0.9, None)


def test_equal_score_is_refused():
    assert not should_promote(0.9, 0.9)


def test_worse_is_refused():
    assert not should_promote(0.85, 0.9)


def test_strictly_better_is_promoted():
    assert should_promote(0.91, 0.9)


def _setup(root, f1, content):
    (root / "reports").mkdir(exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    (root / "reports/metrics.json").write_text(json.dumps({"val_f1": f1}))
    joblib.dump(content, root / "models/candidate.pkl")


def test_first_promotion_creates_model_and_reference(tmp_path):
    _setup(tmp_path, 0.9, "A")
    assert promote("full", root=tmp_path)
    assert (tmp_path / "models/model.pkl").exists()
    assert json.loads((tmp_path / "reports/production_full.json").read_text())["val_f1"] == 0.9


def test_equal_candidate_keeps_current_model(tmp_path):
    _setup(tmp_path, 0.9, "A")
    promote("full", root=tmp_path)
    _setup(tmp_path, 0.9, "B")
    assert not promote("full", root=tmp_path)
    assert joblib.load(tmp_path / "models/model.pkl") == "A"


def test_better_candidate_replaces_model(tmp_path):
    _setup(tmp_path, 0.9, "A")
    promote("full", root=tmp_path)
    _setup(tmp_path, 0.95, "B")
    assert promote("full", root=tmp_path)
    assert joblib.load(tmp_path / "models/model.pkl") == "B"


def test_tags_are_independent(tmp_path):
    _setup(tmp_path, 0.9, "A")
    promote("full", root=tmp_path)
    assert promote("ci", root=tmp_path)   # le tag ci n'a pas encore de référence
    assert (tmp_path / "models/model_ci.pkl").exists()


def test_promotion_exports_features_json(tmp_path, fitted_pipe):
    _setup(tmp_path, 0.9, fitted_pipe)
    promote("full", root=tmp_path)
    feats = json.loads((tmp_path / "models/features.json").read_text())
    assert "Duration" in feats["columns"]
    assert "class3" not in feats["columns"]
    assert "Scr_IP" not in feats["columns"]
    assert feats["model_card"]["metrics"]["val_f1"] == 0.9