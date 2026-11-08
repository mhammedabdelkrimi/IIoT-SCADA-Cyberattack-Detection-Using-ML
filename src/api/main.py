"""API FastAPI : /health, /predict + tableau de bord (/dashboard)."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Union

import joblib
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, field_validator

from src.features import CAT, DROP, clean_frame   # importe aussi src.features (nécessaire au chargement du modèle)

load_dotenv()   # lit .env ; ne remplace pas les variables déjà définies

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))
FEATURES_PATH = Path(os.getenv("FEATURES_PATH", "models/features.json"))
LOG_PATH = Path(os.getenv("PROD_LOG_PATH", "data/production/requests.csv"))
PARAMS_PATH = Path("configs/params.yaml")
DRIFT_SUMMARY = Path("reports/drift_summary.json")
DRIFT_REPORT = Path("reports/drift_report.html")
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="IDS - détection d'attaques")
_model = None


def validation_status() -> int:
    return int(os.getenv("VALIDATION_STATUS", "422"))


def get_model():
    global _model
    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    return _model


def _require_model():
    pipe = get_model()
    if pipe is None:
        raise HTTPException(status_code=503, detail="Modèle de production absent")
    return pipe


def _features_payload() -> dict:
    if FEATURES_PATH.exists():
        try:
            return json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return {}


def known_columns(pipe) -> set:
    """Colonnes acceptées : celles du modèle + celles de features.json (si le fichier existe)."""
    cols = set(pipe.named_steps["clean"].columns_) | set(DROP)
    cols |= set(_features_payload().get("columns", []))
    return cols


def _validate(flow, pipe) -> None:
    known = known_columns(pipe)
    errors = []
    for k, v in flow.data.items():
        if k not in known:
            errors.append(f"colonne inconnue : {k}")
        elif k not in CAT and isinstance(v, str) and v not in ("-", "TRUE", "FALSE", "True", "False"):
            try:
                float(v)
            except ValueError:
                errors.append(f"{k} doit être numérique (reçu : {v!r})")
    if errors:
        raise HTTPException(status_code=validation_status(), detail=errors)


def feature_names(pipe) -> list:
    prep = pipe.named_steps["prep"]
    return list(prep.cont_) + list(prep.pass_) + list(prep.ohe_.get_feature_names_out(prep.cat_))


class Flow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    data: Dict[str, Union[float, str, bool, None]]

    @field_validator("data")
    @classmethod
    def not_empty(cls, v):
        if not v:
            raise ValueError("data ne peut pas être vide")
        return v


@app.exception_handler(RequestValidationError)
async def validation_handler(request, exc):
    errs = [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()]
    return JSONResponse(status_code=validation_status(),
                        content={"detail": "Données mal formatées", "errors": errs})


# ---------------------------------------------------------------- API de prédiction
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": get_model() is not None}


@app.post("/predict")
def predict(flow: Flow):
    pipe = _require_model()
    _validate(flow, pipe)

    X = pd.DataFrame([flow.data])
    proba = float(pipe.predict_proba(X)[0, 1])
    label = "Attack" if proba >= 0.5 else "Normal"

    row = pipe.named_steps["clean"].transform(X)   # journal pour le monitoring et le tableau de bord
    row["prediction"] = label
    row["probability_attack"] = round(proba, 4)
    row["received_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)

    return {"prediction": label, "probability_attack": round(proba, 4)}


# ---------------------------------------------------------------- tableau de bord
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/dashboard")


@app.get("/dashboard", include_in_schema=False)
def dashboard():
    return FileResponse(STATIC_DIR / "dashboard.html", media_type="text/html")


@app.get("/model-info")
def model_info():
    """Colonnes d'entrée + fiche de performance du modèle en production."""
    pipe = get_model()
    payload = _features_payload()
    cols = list(pipe.named_steps["clean"].columns_) if pipe is not None else payload.get("columns", [])
    return {"model_loaded": pipe is not None, "columns": cols, "categorical": CAT,
            "card": payload.get("model_card", {}), "validation_status": validation_status()}


@app.post("/explain")
def explain(flow: Flow, top: int = 8):
    """Contribution de chaque variable à la décision (régression logistique : coefficient x valeur transformée)."""
    pipe = _require_model()
    _validate(flow, pipe)
    X = pd.DataFrame([flow.data])
    v = pipe.named_steps["prep"].transform(pipe.named_steps["clean"].transform(X))[0]
    clf = pipe.named_steps["clf"]
    contrib = v * clf.coef_[0]
    names = feature_names(pipe)
    order = np.argsort(-np.abs(contrib))[:max(1, top)]
    return {"intercept": round(float(clf.intercept_[0]), 4),
            "contributions": [{"feature": names[i],
                               "contribution": round(float(contrib[i]), 4),
                               "direction": "attack" if contrib[i] > 0 else "normal"}
                              for i in order]}


@app.get("/sample")
def sample(kind: str = "attack"):
    """Un flux réel tiré au hasard du jeu de validation (pour la démonstration)."""
    if kind not in ("attack", "normal"):
        raise HTTPException(status_code=422, detail="kind doit valoir 'attack' ou 'normal'")
    p = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8"))
    path = Path(p["data"]["val_csv"])
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Jeu de validation introuvable : {path}")
    d = p["data"]
    df = pd.read_csv(path, low_memory=False)
    mask = df[d["target"]] == d["positive"]
    df = df[mask] if kind == "attack" else df[~mask]
    if df.empty:
        raise HTTPException(status_code=404, detail="Aucun flux de ce type dans le jeu de validation")
    row = df.sample(1).iloc[0]
    cleaned = clean_frame(pd.DataFrame([row]))
    return {"data": json.loads(cleaned.to_json(orient="records"))[0],
            "true_label": str(row[d["target"]]),
            "true_type": str(row[d["type_column"]]) if d.get("type_column") in df.columns else None}


@app.get("/stats")
def stats(limit: int = 15):
    """Statistiques sur les requêtes reçues (journal PROD_LOG_PATH) + dernier rapport de dérive."""
    drift = {"available": False}
    if DRIFT_SUMMARY.exists():
        try:
            drift = {"available": True, **json.loads(DRIFT_SUMMARY.read_text(encoding="utf-8")),
                     "report": DRIFT_REPORT.exists()}
        except ValueError:
            pass
    empty = {"total": 0, "attacks": 0, "normal": 0, "attack_rate": None,
             "histogram": [0] * 10, "recent": [], "drift": drift}
    if not LOG_PATH.exists():
        return empty
    df = pd.read_csv(LOG_PATH)
    if df.empty or "prediction" not in df.columns:
        return empty
    if "received_at" not in df.columns:
        df["received_at"] = None
    attacks = int((df["prediction"] == "Attack").sum())
    hist = np.histogram(df["probability_attack"].astype(float), bins=10, range=(0, 1))[0]
    keep = [c for c in ["received_at", "prediction", "probability_attack", "Protocol", "Service",
                        "Duration", "Scr_bytes"] if c in df.columns]
    recent = df.tail(max(1, limit))[keep].iloc[::-1]
    return {"total": int(len(df)), "attacks": attacks, "normal": int(len(df) - attacks),
            "attack_rate": round(attacks / len(df), 4), "histogram": hist.tolist(),
            "recent": json.loads(recent.to_json(orient="records")), "drift": drift}


@app.get("/drift-report", include_in_schema=False)
def drift_report():
    if not DRIFT_REPORT.exists():
        raise HTTPException(status_code=404, detail="Aucun rapport de dérive : lancez python -m src.monitoring")
    return FileResponse(DRIFT_REPORT, media_type="text/html")