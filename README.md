# IIoT / SCADA – Détection d'intrusion par Machine Learning (MLOps)

Système de détection d'attaques réseau pour environnements industriels (IIoT / SCADA), basé sur le jeu de données **X-IIoTID**. Chaque flux réseau est classé **Normal** ou **Attack** par une régression logistique, expliqué variable par variable, exposé par une API, puis surveillé en production (dérive des données).

Le projet couvre toute la chaîne MLOps : données versionnées, entraînement reproductible, suivi des expériences, promotion contrôlée du modèle, API, conteneurs, intégration continue et monitoring.

---

## Sommaire

1. [Aperçu](#aperçu)
2. [Résultats](#résultats)
3. [Architecture](#architecture)
4. [Structure du dépôt](#structure-du-dépôt)
5. [Installation](#installation)
6. [Utilisation](#utilisation)
7. [API](#api)
8. [Tableau de bord](#tableau-de-bord)
9. [Monitoring de la dérive](#monitoring-de-la-dérive)
10. [Docker](#docker)
11. [Intégration continue (GitHub Actions)](#intégration-continue-github-actions)
12. [Tests](#tests)
13. [Limites](#limites)
14. [Pistes d'amélioration](#pistes-damélioration)

---

## Aperçu

| Élément | Choix |
|---|---|
| Problème | Classification binaire de flux réseau : `Normal` / `Attack` |
| Données | X-IIoTID (jeu d'entraînement de 50 000 lignes, jeu de validation de 10 000 lignes) |
| Modèle | Régression logistique (scikit-learn), recherche d'hyperparamètres par `GridSearchCV` |
| Versionnement des données | DVC, stockage distant Backblaze B2 (compatible S3) |
| Suivi des expériences | MLflow (paramètres, métriques, courbes d'apprentissage, registre de modèles) |
| API | FastAPI + Pydantic |
| Monitoring | Evidently (tests KS et chi²) |
| Conteneurs | Docker et Docker Compose |
| CI | GitHub Actions (tests, entraînement, construction de l'image) |

---

## Résultats

Scores du modèle en production, sur le jeu de **validation** indépendant (classe positive : `Attack`) :

| Métrique | Valeur |
|---|---|
| F1 | 0,9750 |
| Exactitude | 0,9760 |
| Précision | 0,9880 |
| Rappel | 0,9624 |
| AUC | 0,9933 |

Hyperparamètres retenus : `C = 10`, sans pondération de classes.

Rappel par famille d'attaque (jeu de test) :

| Famille | Attaques testées | Rappel |
|---|---|---|
| MitM | 1 | 0,0 % |
| fuzzing | 13 | 0,0 % |
| C&C | 25 | 48,0 % |
| TCP Relay | 27 | 51,9 % |
| Discovering_resources | 291 | 58,4 % |
| Modbus_register_reading | 77 | 84,4 % |
| Generic_scanning | 648 | 99,5 % |
| RDOS | 1712 | 99,8 % |
| BruteForce, Dictionary, Exfiltration, False_data_injection, MQTT_cloud_broker_subscription, Reverse_shell, Scanning_vulnerability, insider_malcious | 9 à 638 | 100 % |

Le modèle détecte presque tout le volume d'attaques, mais rate les familles rares ou proches du trafic normal. Avec un seul exemple de MitM dans le jeu de test, le 0 % n'est pas significatif statistiquement.

---

## Architecture

```
données (DVC + Backblaze) ──► train.py ──► candidate.pkl + metrics.json + model_card.json
                                  │
                                  └──► MLflow (paramètres, métriques, courbes, modèle)

candidate.pkl ──► promote.py ──(F1 de validation strictement meilleur)──► model.pkl + features.json

model.pkl ──► API FastAPI ──► journal des requêtes ──► monitoring Evidently ──► rapport de dérive
                  │
                  └──► tableau de bord (/dashboard)
```

Règle de promotion : un nouveau modèle ne remplace celui en production que si son `val_f1` est **strictement supérieur**. Le premier modèle est promu automatiquement.

---

## Structure du dépôt

```
.
├── configs/
│   └── params.yaml            # tous les réglages (données, split, grille, MLflow, monitoring)
├── data/
│   └── raw/                   # jeux de données (suivis par DVC, non versionnés dans Git)
├── models/                    # candidate.pkl, model.pkl, features.json
├── reports/                   # metrics.json, model_card.json, grid_search.csv, recall_par_type.csv
├── scripts/                   # scripts utilitaires (démarrage MLflow, diagnostic)
├── src/
│   ├── features.py            # nettoyage et pipeline scikit-learn
│   ├── train.py               # entraînement + suivi MLflow
│   ├── promote.py             # promotion contrôlée du modèle
│   ├── monitoring.py          # rapport de dérive (Evidently)
│   ├── simulate_traffic.py    # génération de trafic de démonstration
│   └── api/
│       ├── main.py            # API FastAPI
│       └── static/dashboard.html
├── tests/                     # tests pytest
├── dvc.yaml                   # définition du pipeline
├── Dockerfile                 # cibles : api, monitor
├── docker-compose.yml
├── requirements.txt           # environnement de développement complet
├── requirements-api.txt       # dépendances de l'image API
├── requirements-monitor.txt   # dépendances de l'image monitoring
├── .env.example               # modèle de configuration (sans secret)
└── pytest.ini
```

---

## Installation

Prérequis : Python 3.11, Git, Docker Desktop (optionnel), un accès au stockage DVC.

```powershell
git clone https://github.com/mhammedabdelkrimi/IIoT-SCADA-Cyberattack-Detection-Using-ML.git
cd IIoT-SCADA-Cyberattack-Detection-Using-ML

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Sous Linux ou macOS : `source .venv/bin/activate`.

### Configuration

Copiez le modèle de configuration puis renseignez vos valeurs :

```powershell
Copy-Item .env.example .env
```

Le fichier `.env` contient notamment l'adresse MLflow et les clés Backblaze. **Il ne doit jamais être versionné** (il est dans `.gitignore`).

Pour DVC, déclarez vos clés localement, sans les écrire dans un fichier suivi par Git :

```powershell
dvc remote modify --local <nom_du_remote> access_key_id <keyID>
dvc remote modify --local <nom_du_remote> secret_access_key <applicationKey>
dvc pull
```

---

## Utilisation

Toutes les commandes se lancent depuis la racine, avec l'environnement virtuel activé.

### 1. Entraîner

MLflow doit être joignable à l'adresse de `MLFLOW_TRACKING_URI`. Sans serveur, commentez cette ligne dans `.env` : MLflow écrit alors dans `mlflow.db` en local.

```powershell
# optionnel : serveur MLflow dans un second terminal
.\scripts\start_mlflow.ps1

dvc repro
```

### 2. Promouvoir le modèle

```powershell
python -m src.promote --tag full
```

L'option `--force` régénère le modèle de production et sa fiche même sans amélioration du score :

```powershell
python -m src.promote --tag full --force
```

### 3. Lancer l'API

```powershell
uvicorn src.api.main:app --port 8000
```

Le tableau de bord est sur `http://localhost:8000/dashboard`, la documentation interactive sur `http://localhost:8000/docs`.

### 4. Visualiser les expériences MLflow

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## API

| Méthode | Chemin | Rôle |
|---|---|---|
| GET | `/health` | État du service et du chargement du modèle |
| POST | `/predict` | Prédiction pour un flux (chaque requête est journalisée) |
| POST | `/explain` | Contribution de chaque variable à la décision |
| GET | `/model-info` | Colonnes d'entrée et fiche de performance du modèle |
| GET | `/stats` | Statistiques des requêtes reçues et état de la dérive |
| GET | `/sample` | Flux réel tiré du jeu de validation (`?kind=attack` ou `normal`) |
| GET | `/drift-report` | Rapport de dérive détaillé (HTML) |
| GET | `/dashboard` | Tableau de bord |

Exemple d'appel :

```powershell
curl.exe -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@examples/predict_ok.json"
```

Corps de la requête :

```json
{"data": {"Protocol": "tcp", "Service": "http", "Duration": 0.5, "Scr_bytes": 120, "Des_bytes": 300, "Scr_pkts": 3, "Des_pkts": 4, "anomaly_alert": "FALSE"}}
```

Réponse :

```json
{"prediction": "Normal", "probability_attack": 0.0123}
```

Les données mal formées (colonne inconnue, valeur non numérique, corps vide) sont refusées avec un code **422**. Le code est configurable avec la variable `VALIDATION_STATUS` (par exemple `400`).

Une requête est classée `Attack` si la probabilité d'attaque est supérieure ou égale à 0,5.

---

## Tableau de bord

Page unique servie par l'API, sans dépendance externe. Thème sombre (noir et rouge) et thème clair (bleu et blanc), au choix.

| Page | Contenu |
|---|---|
| Vue d'ensemble | Scores de validation, trafic reçu, radar de performance, familles d'attaques, diagnostic des données |
| Analyse d'un flux | Formulaire de saisie, chargement d'un flux réel, verdict, jauge de probabilité, variables déterminantes |
| Catégories d'attaques | Répartition, taux de détection et fiche par famille d'attaque |
| Performance du modèle | Métriques train/test/validation, courbes d'apprentissage, rappel par famille |
| Surveillance | Distribution des scores, évolution des requêtes, état de la dérive, journal |
| Architecture MLOps | Chaîne de production et points d'accès de l'API |

Les graphiques proviennent de la fiche du modèle (`features.json`, clé `model_card`) et du journal des requêtes. La carte « État des données » de la vue d'ensemble indique ce qui manque, le cas échéant.

---

## Monitoring de la dérive

Les requêtes reçues par l'API sont enregistrées dans `data/production/requests.csv`. Le script de monitoring les compare aux données d'entraînement avec Evidently :

```powershell
python -m src.simulate_traffic --n 100          # trafic normal
python -m src.monitoring

python -m src.simulate_traffic --n 100 --drift  # trafic déformé
python -m src.monitoring
```

- Il faut au moins `min_requests` requêtes (50 par défaut, dans `params.yaml`).
- Une dérive est signalée quand plus de `drift_share` des colonnes dérivent (15 % par défaut).
- Sorties : `reports/drift_summary.json` et `reports/drift_report.html`.

Le trafic produit par `simulate_traffic.py` est **simulé** à partir du jeu de validation. Il sert à démontrer le monitoring, pas à représenter un trafic de production réel.

---

## Docker

Deux services construits à partir d'un même `Dockerfile` :

| Service | Rôle |
|---|---|
| `api` | API FastAPI et tableau de bord sur le port 8000 |
| `monitor` | Régénère le rapport de dérive chaque semaine |

```powershell
docker compose up --build -d
docker compose ps
curl.exe http://localhost:8000/health
```

`docker compose ps` doit montrer `api` en `healthy`. Les volumes montent `models/`, `reports/` et `data/` pour que le modèle, la fiche et les journaux restent en dehors de l'image.

Les dépendances sont séparées par service (`requirements-api.txt`, `requirements-monitor.txt`) pour garder des images légères.

---

## Intégration continue (GitHub Actions)

Le workflow `ci-cd` comporte trois jobs :

| Job | Rôle |
|---|---|
| `test` | Lance `pytest` |
| `train` | Récupère les données avec `dvc pull`, entraîne, puis promeut le modèle |
| `build` | Construit l'image Docker si un modèle a été promu |

### Secrets à définir

Dans GitHub : **Settings → Secrets and variables → Actions → New repository secret**.

| Nom | Contenu |
|---|---|
| `B2_KEY_ID` | `keyID` Backblaze utilisé par DVC |
| `B2_APPLICATION_KEY` | `applicationKey` correspondante |

Sans ces secrets, `dvc pull` échoue avec « Unable to locate credentials ».

La CI peut committer ses propres fichiers sur `main`. Faites donc `git pull --rebase` avant chaque `git push`.

---

## Tests

```powershell
pytest -q
```

Les tests couvrent l'entraînement de bout en bout, la promotion du modèle, la validation de l'API et la page du tableau de bord. Les avertissements de dépréciation de `mlflow`, `pydantic` et `fastapi` sont sans effet sur le projet.

---

## Limites

- Le découpage apprentissage/test est aléatoire dans un même jeu de données : les flux d'une même attaque se ressemblent, donc les scores de test sont plutôt optimistes. Le **jeu de validation indépendant** est la mesure de référence.
- Un modèle linéaire capture mal les familles rares ou proches du trafic normal (MitM, fuzzing, C&C, TCP Relay).
- Les scores décrivent le passé : une évolution du trafic dégrade la détection, d'où la surveillance de la dérive.
- Le trafic de production du monitoring est simulé.
- L'explication par variable (coefficient × valeur transformée) n'est exacte que pour un modèle linéaire.
- L'envoi des artefacts MLflow vers Backblaze n'est pas validé. Les données et le modèle sont versionnés par DVC ; le suivi MLflow fonctionne en local.
- Un IDS complète les autres défenses (segmentation, signatures, journalisation) et ne les remplace pas. Une alerte doit être qualifiée par un analyste avant toute action sur un procédé industriel.

---

## Pistes d'amélioration

- Pondérer les classes ou rééquilibrer les familles rares.
- Tester un modèle non linéaire (arbres boostés) et comparer au modèle linéaire sur la validation.
- Abaisser le seuil de décision pour gagner en rappel, au prix de davantage de fausses alertes.
- Évaluer sur un jeu de données externe pour mesurer la généralisation.
- Brancher le monitoring sur un trafic réel et déclencher automatiquement un réentraînement en cas de dérive.
- Réparer l'envoi des artefacts MLflow vers Backblaze (clé et endpoint à aligner sur ceux de DVC).

---

## Auteur

Projet réalisé par Mhammed Abdelkrimi dans le cadre d'un projet MLOps de détection d'intrusion IIoT / SCADA.
