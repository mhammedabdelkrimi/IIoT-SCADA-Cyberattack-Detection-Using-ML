"""Logique de prédiction et journalisation des requêtes (sans dépendance à FastAPI)."""
import csv
import threading
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.features import FEATURES, PARAMS, clean_frame

_LOCK = threading.Lock()
LOG_COLUMNS = ["timestamp"] + FEATURES + ["prediction", "probability_attack"]


def load_model(path):
    return joblib.load(path)


def predict_one(model, features: dict):
    """Retourne (étiquette, probabilité d'attaque) pour une ligne."""
    frame = clean_frame(pd.DataFrame([features]))
    proba = float(model.predict_proba(frame)[0, 1])
    label = PARAMS["data"]["positive_label"] if proba >= 0.5 else "Normal"
    return label, proba


def log_prediction(log_path, features: dict, label: str, proba: float):
    """Ajoute la requête au fichier de logs (sert au monitoring de dérive). Ne bloque jamais l'API."""
    try:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {c: features.get(c) for c in FEATURES}
        row.update(timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   prediction=label, probability_attack=round(proba, 6))
        with _LOCK:
            new_file = not path.exists()
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
    except Exception:
        pass