from pathlib import Path

import pytest
import yaml
from conftest import make_df

PARAMS = Path(__file__).resolve().parents[1] / "configs" / "params.yaml"


def _prepare(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "data/raw").mkdir(parents=True)
    p = yaml.safe_load(PARAMS.read_text(encoding="utf-8"))
    p["data"]["train_csv"] = "data/raw/train.csv"
    p["data"]["val_csv"] = "data/raw/val.csv"
    p["model"].update({"C_grid": [1, 100], "class_weight_grid": [None], "cv_folds": 2})
    (tmp_path / "configs/params.yaml").write_text(yaml.safe_dump(p), encoding="utf-8")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    return p


def test_train_end_to_end(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    make_df(1500, 0)[0].to_csv("data/raw/train.csv", index=False)
    make_df(400, 1)[0].to_csv("data/raw/val.csv", index=False)
    from src import train
    train.main()
    assert (tmp_path / "models/candidate.pkl").exists()
    assert "val_f1" in (tmp_path / "reports/metrics.json").read_text()


def test_missing_csv_gives_clear_message(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    from src import train
    with pytest.raises(SystemExit) as e:
        train.main()
    assert "introuvable" in str(e.value)