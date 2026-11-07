"""Décide si le modèle candidat remplace le modèle en production."""
import argparse
import json
import os
import shutil
from pathlib import Path

import joblib
from dotenv import load_dotenv

from src.features import CAT, DROP

load_dotenv()   # lit .env (MODEL_PATH, FEATURES_PATH...) ; ne remplace pas les variables déjà définies


def tagged(path: Path, tag: str) -> Path:
    """full -> chemin tel quel (models/model.pkl) ; ci -> models/model_ci.pkl"""
    return path if tag == "full" else path.with_name(f"{path.stem}_{tag}{path.suffix}")


def model_path(tag: str) -> Path:
    return tagged(Path(os.getenv("MODEL_PATH", "models/model.pkl")), tag)


def features_path(tag: str) -> Path:
    return tagged(Path(os.getenv("FEATURES_PATH", "models/features.json")), tag)


def should_promote(candidate_f1: float, production_f1) -> bool:
    """Premier modèle : promu. Ensuite : STRICTEMENT supérieur uniquement."""
    return production_f1 is None or candidate_f1 > production_f1


def export_features(model_file: Path, features_file: Path) -> None:
    """Écrit la liste exacte des colonnes d'entrée du modèle promu (contrat d'entrée de l'API)."""
    try:
        clean = joblib.load(model_file).named_steps["clean"]
        payload = {"columns": clean.columns_, "categorical": CAT, "dropped": DROP}
    except (AttributeError, KeyError):
        print("features non exportées : modèle non reconnu")
        return
    features_file.parent.mkdir(parents=True, exist_ok=True)
    features_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def promote(tag: str, root: str = ".") -> bool:
    root = Path(root)
    metrics = json.loads((root / "reports/metrics.json").read_text(encoding="utf-8"))
    ref = root / f"reports/production_{tag}.json"
    prod = json.loads(ref.read_text(encoding="utf-8"))["val_f1"] if ref.exists() else None
    cand = metrics["val_f1"]
    ok = should_promote(cand, prod)

    print(f"[{tag}] val_f1 candidat = {cand} | val_f1 production = {prod}")
    if ok:
        dest = root / model_path(tag)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / "models/candidate.pkl", dest)
        ref.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        export_features(dest, root / features_path(tag))
        print(f"=> PROMU : le candidat devient le modèle de production ({dest}).")
    else:
        print("=> REFUSÉ : modèle actuel conservé.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full", help="full = votre PC, ci = GitHub Actions")
    args = ap.parse_args()
    ok = promote(args.tag)
    out = os.getenv("GITHUB_OUTPUT")          # lu par la CI pour décider du build
    if out:
        with open(out, "a") as f:
            f.write(f"promoted={'true' if ok else 'false'}\n")


if __name__ == "__main__":
    main()