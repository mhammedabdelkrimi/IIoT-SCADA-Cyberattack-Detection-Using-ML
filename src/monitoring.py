"""Rapport de dérive (Evidently) : requêtes reçues vs données d'entraînement."""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.features import CAT, clean_frame

load_dotenv()   # lit .env (PROD_LOG_PATH...) ; ne remplace pas les variables déjà définies


def load_current(log_path, min_requests):
    """Requêtes journalisées, ou None s'il y en a trop peu (ou aucune)."""
    log = Path(log_path)
    if not log.exists():
        return None
    cur = pd.read_csv(log).drop(columns=["prediction", "probability_attack"], errors="ignore")
    if len(cur) < min_requests:
        return None
    return cur


def analyze(ref, cur, drift_share, html_path=None):
    cols = [c for c in ref.columns if c in cur.columns]
    report = Report(metrics=[DataDriftPreset(
        drift_share=drift_share, num_stattest="ks", cat_stattest="chisquare")])
    report.run(reference_data=ref[cols], current_data=cur[cols])
    if html_path:
        report.save_html(str(html_path))
    res = report.as_dict()["metrics"][0]["result"]
    return {"dataset_drift": bool(res["dataset_drift"]),
            "share_drifted_columns": res["share_of_drifted_columns"],
            "n_drifted_columns": res["number_of_drifted_columns"],
            "n_requests": len(cur)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate-drift", action="store_true",
                    help="fabrique des requêtes décalées (démo sans API)")
    args = ap.parse_args()

    p = yaml.safe_load(open("configs/params.yaml", encoding="utf-8"))
    m = p["monitoring"]

    ref = clean_frame(pd.read_csv(p["data"]["train_csv"], low_memory=False)
                      .sample(m["reference_sample"], random_state=42))

    if args.simulate_drift:
        cur = ref.sample(200, random_state=1).copy()
        cols = [c for c in cur.columns if c not in CAT and cur[c].nunique() > 2][:10]
        cur[cols] = cur[cols] * 5
    else:
        log_path = os.getenv("PROD_LOG_PATH", "data/production/requests.csv")
        cur = load_current(log_path, m["min_requests"])
        if cur is None:
            print(f"Pas assez de requêtes dans {log_path} (minimum {m['min_requests']}) : pas de rapport.")
            return
        cur = clean_frame(cur)

    Path("reports").mkdir(exist_ok=True)
    summary = analyze(ref, cur, m["drift_share"], "reports/drift_report.html")
    Path("reports/drift_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary)
    if summary["dataset_drift"]:
        print("ATTENTION : dérive détectée, envisagez un réentraînement.")


if __name__ == "__main__":
    main()