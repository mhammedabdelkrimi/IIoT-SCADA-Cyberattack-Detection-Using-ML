"""Validation des requêtes avec Pydantic (aucune dépendance à FastAPI : testable seule)."""
from typing import Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.features import CATEGORICAL, FEATURES, NUMERIC

Value = Union[float, str, bool, None]


class PredictRequest(BaseModel):
    """Une ligne de trafic réseau. Les colonnes absentes ou nulles sont imputées par le modèle."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "features": {
                    "Protocol": "tcp", "Service": "http", "Duration": 0.5, "Scr_bytes": 320,
                    "Des_bytes": 1200, "Conn_state": 1, "anomaly_alert": 0,
                }
            }
        },
    )

    features: Dict[str, Value] = Field(..., description="Nom de colonne -> valeur")

    @field_validator("features")
    @classmethod
    def check_features(cls, feats):
        if not feats:
            raise ValueError("« features » ne peut pas être vide")
        unknown = sorted(set(feats) - set(FEATURES))
        if unknown:
            raise ValueError(f"colonnes inconnues : {unknown}")
        for name, value in feats.items():
            if value is None:
                continue
            if name in NUMERIC and (isinstance(value, str)):
                raise ValueError(f"« {name} » doit être numérique (reçu : {value!r})")
            if name in CATEGORICAL and not isinstance(value, str):
                raise ValueError(f"« {name} » doit être un texte (reçu : {value!r})")
        return feats


class PredictResponse(BaseModel):
    prediction: str
    probability_attack: float
    model_loaded_from: Optional[str] = None