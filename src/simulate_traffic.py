"""Envoie des lignes de data_val_10000.csv à l'API (option --drift : valeurs décalées)."""
import argparse
import json

import httpx
import pandas as pd
import yaml

from src.features import CAT, clean_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--url", default="http://localhost:8000/predict")
    args = ap.parse_args()

    p = yaml.safe_load(open("configs/params.yaml", encoding="utf-8"))
    X = clean_frame(pd.read_csv(p["data"]["val_csv"], low_memory=False)
                    .sample(args.n, random_state=1))
    if args.drift:
        cols = [c for c in X.columns if c not in CAT and X[c].nunique() > 2][:10]
        X[cols] = X[cols] * 5
    ok = 0
    for row in json.loads(X.to_json(orient="records")):
        ok += httpx.post(args.url, json={"data": row}, timeout=30).status_code == 200
    print(f"{ok}/{args.n} requêtes acceptées")


if __name__ == "__main__":
    main()