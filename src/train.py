"""Entraînement + suivi MLflow."""
import json
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (accuracy_score, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from src.features import build_pipeline

load_dotenv()   # lit .env ; ne remplace pas les variables déjà définies

# Backblaze B2 refuse les sommes de contrôle ajoutées par défaut par boto3 >= 1.36
# (erreur "BadStatusLine" / "Connection was closed" à l'envoi des artefacts MLflow).
os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")
os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_required")

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_params(path="configs/params.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_xy(df, p):
    y = (df[p["data"]["target"]] == p["data"]["positive"]).astype(int)
    return df.drop(columns=[p["data"]["target"]]), y   # class1/2 retirées par le Cleaner


def evaluate(pipe, X, y, prefix):
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]
    return {
        f"{prefix}_accuracy": accuracy_score(y, pred),
        f"{prefix}_precision": precision_score(y, pred, zero_division=0),
        f"{prefix}_recall": recall_score(y, pred, zero_division=0),
        f"{prefix}_f1": f1_score(y, pred, zero_division=0),
        f"{prefix}_auc": roc_auc_score(y, proba),
        f"{prefix}_loss": log_loss(y, proba),
    }


def recall_per_type(pipe, X, y, type_col):
    """Part des attaques détectées, pour chaque type d'attaque (colonne class1)."""
    if type_col not in X.columns:
        return pd.DataFrame()
    pred = pipe.predict(X)
    d = pd.DataFrame({"type": X[type_col].values, "y": y.values, "pred": pred})
    d = d[d["y"] == 1]
    out = d.groupby("type").agg(n=("y", "size"), recall=("pred", "mean")).reset_index()
    return out.sort_values("recall")


def main():
    p = load_params()
    d, mp = p["data"], p["model"]
    for f in (d["train_csv"], d["val_csv"]):
        if not Path(f).exists():
            raise SystemExit(f"Fichier de données introuvable : {f}  (essayez 'dvc pull')")

    X, y = split_xy(pd.read_csv(d["train_csv"], low_memory=False), p)
    Xv, yv = split_xy(pd.read_csv(d["val_csv"], low_memory=False), p)
    stratif = y if y.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=p["split"]["test_size"],
        random_state=p["split"]["random_state"], stratify=stratif)

    gs = GridSearchCV(
        build_pipeline(max_iter=mp["max_iter"], tol=mp["tol"]),
        {"clf__C": mp["C_grid"], "clf__class_weight": mp["class_weight_grid"]},
        cv=StratifiedKFold(mp["cv_folds"], shuffle=True, random_state=42),
        scoring="f1", n_jobs=-1, return_train_score=True, verbose=1)
    gs.fit(Xtr, ytr)
    best = gs.best_estimator_
    print("Meilleurs paramètres :", gs.best_params_)

    metrics = {}
    metrics.update(evaluate(best, Xtr, ytr, "train"))
    metrics.update(evaluate(best, Xte, yte, "test"))
    metrics.update(evaluate(best, Xv, yv, "val"))
    metrics = {k: round(float(v), 6) for k, v in metrics.items()}

    Path("reports").mkdir(exist_ok=True)
    grid_df = pd.DataFrame(gs.cv_results_)[["param_clf__C", "param_clf__class_weight",
                                            "mean_train_score", "mean_test_score"]]
    grid_df.to_csv("reports/grid_search.csv", index=False)
    rec = recall_per_type(best, Xte, yte, d.get("type_column", "class1"))
    rec.to_csv("reports/recall_par_type.csv", index=False)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", p["mlflow"]["experiment"]))
    with mlflow.start_run():
        mlflow.log_params({
            "C": gs.best_params_["clf__C"],
            "class_weight": str(gs.best_params_["clf__class_weight"]),
            "max_iter": mp["max_iter"], "tol": mp["tol"],
            "n_train": len(Xtr), "n_val": len(Xv)})
        mlflow.log_metrics(metrics)

        # Courbes d'apprentissage : loss, accuracy, F1 selon la part du Train utilisée
        lc = {"steps": []}
        for frac in np.linspace(0.1, 1.0, 10):
            n = int(len(Xtr) * frac)
            m = clone(best).fit(Xtr.iloc[:n], ytr.iloc[:n])
            step = int(round(frac * 100))
            lc["steps"].append(step)
            for name, Xs, ys in (("train", Xtr.iloc[:n], ytr.iloc[:n]), ("test", Xte, yte)):
                proba = m.predict_proba(Xs)[:, 1]
                pred = m.predict(Xs)
                vals = {"loss": log_loss(ys, proba, labels=[0, 1]),
                        "accuracy": accuracy_score(ys, pred),
                        "f1": f1_score(ys, pred, zero_division=0)}
                for k, v in vals.items():
                    mlflow.log_metric(f"lc_{name}_{k}", v, step=step)
                    lc.setdefault(f"{name}_{k}", []).append(round(float(v), 6))

        mlflow.log_artifact("reports/grid_search.csv")
        mlflow.log_artifact("reports/recall_par_type.csv")
        mlflow.sklearn.log_model(best, "model",
                                 registered_model_name=p["mlflow"]["registered_model"])

    card = {   # lu par promote.py puis affiché dans le tableau de bord
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "params": {"C": gs.best_params_["clf__C"],
                   "class_weight": gs.best_params_["clf__class_weight"],
                   "max_iter": mp["max_iter"], "tol": mp["tol"], "cv_folds": mp["cv_folds"]},
        "sizes": {"train": len(Xtr), "test": len(Xte), "validation": len(Xv)},
        "learning_curves": lc,
        "recall_par_type": json.loads(rec.to_json(orient="records")),
    }
    Path("reports/model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    Path("models").mkdir(exist_ok=True)
    joblib.dump(best, "models/candidate.pkl")
    Path("reports/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()