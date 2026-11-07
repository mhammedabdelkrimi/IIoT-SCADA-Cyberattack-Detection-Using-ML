import numpy as np
import pandas as pd
import pytest

from src.features import build_pipeline


def make_df(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({
        "Scr_IP": "1.1.1.1", "Des_port": 80,
        "class1": np.where(y == 1, "scan", "Normal"),
        "class2": np.where(y == 1, "recon", "Normal"),
        "class3": np.where(y == 1, "Attack", "Normal"),
        "Protocol": rng.choice(["tcp", "udp"], n),
        "Service": rng.choice(["http", "-"], n),
        "Duration": rng.lognormal(0, 2, n) + y * 5,
        "Scr_bytes": rng.lognormal(3, 1, n),
        "anomaly_alert": rng.choice(["TRUE", "FALSE"], n),
    })
    return df, y


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Tests déterministes : indépendants de votre fichier .env."""
    monkeypatch.setenv("MODEL_PATH", "models/model.pkl")
    monkeypatch.setenv("FEATURES_PATH", "models/features.json")
    monkeypatch.delenv("VALIDATION_STATUS", raising=False)


@pytest.fixture
def data():
    return make_df()


@pytest.fixture
def fitted_pipe(data):
    df, y = data
    return build_pipeline().fit(df, y)