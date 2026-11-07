"""API FastAPI : /health et /predict."""
import json
import os
from pathlib import Path
from typing import Dict, Union

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, field_validator

from src.features import CAT, DROP   # importe aussi src.features, nécessaire au chargement du modèle

load_dotenv()   # lit .env ; ne remplace pas les variables déjà définies

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pkl"))
FEATURES_PATH = Path(os.getenv("FEATURES_PATH", "models/features.json"))
LOG_PATH = Path(os.getenv("PROD_LOG_PATH", "data/production/requests.csv"))

app = FastAPI(title="IDS - détection d'attaques")
_model = None


def validation_status() -> int:
    return int(os.getenv("VALIDATION_STATUS", "422"))


def get_model():
    global _model
    if _model is None and MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
    return _model


def known_columns(pipe) -> set:
    """Colonnes acceptées : celles du modèle + celles de features.json (si le fichier existe)."""
    cols = set(pipe.named_steps["clean"].columns_) | set(DROP)
    if FEATURES_PATH.exists():
        try:
            cols |= set(json.loads(FEATURES_PATH.read_text(encoding="utf-8"))["columns"])
        except (ValueError, KeyError):
            pass
    return cols


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


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": get_model() is not None}


@app.post("/predict")
def predict(flow: Flow):
    pipe = get_model()
    if pipe is None:
        raise HTTPException(status_code=503, detail="Modèle de production absent")

    clean = pipe.named_steps["clean"]
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

    X = pd.DataFrame([flow.data])
    proba = float(pipe.predict_proba(X)[0, 1])
    label = "Attack" if proba >= 0.5 else "Normal"

    row = clean.transform(X)                       # journal pour le monitoring
    row["prediction"] = label
    row["probability_attack"] = round(proba, 4)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)

    return {"prediction": label, "probability_attack": round(proba, 4)}