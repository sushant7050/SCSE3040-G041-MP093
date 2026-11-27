
# SCSE3040 - MLOps Mini Project

[![CI](https://github.com/sushant7050/SCSE3040-G041-MP093/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/sushant7050/SCSE3040-G041-MP093/actions/workflows/ci.yml)

## Group Information
- Group ID: G041
- Problem ID: MP093
- Problem: Ration Shop Distribution (Nagpur)

## Group Members

| Name | Enrollment Number |
|---|---|
| Shaurya Nigam | S24CSEU0497 |
| Prasantkumar | S24CSEU2415 |
| Rishabh Jain | S24CSEU2390 |
| Sushant Kumar | S24CSEU2413 |

## Project Overview
Some fair price (ration) shops in Nagpur run out of grain by noon, so families
are sent back empty handed. The client, the Nagpur Food and Civil Supplies
Office (a made-up organisation), wants to know how much grain each shop will
hand out per day so that stock can be sent in time. We predict `grain_kg` per
shop per day. This is a regression problem, measured by MAE (Mean Absolute
Error): the average size of a mistake, in kg.

This repo holds the whole path from raw data to a running service: versioned
data, data checks and cleaning, tracked model training, a prediction API in
Docker with Kubernetes manifests, automated tests, and monitoring with
retraining once newer data arrived.

## Pipeline
1. **Data versioning (DVC):** `data/raw/train.csv.dvc` and `data/raw/later.csv.dvc` pin the exact data files.
2. **Data checks:** `mp093.check_data` looks for the seven kinds of data problem in `train.csv`.
3. **Cleaning:** `clean()` in `mp093.data` fixes the problems found, without writing to the CSV.
4. **Training:** `mp093.train` tracks 4 MLflow runs in `mlflow.db` and registers the best as v1, `@champion`.
5. **API:** `app/main.py` (FastAPI) rejects `working_hours` above 15 and `cards_attached` below 100 with status 422.
6. **Logging:** `mp093.logging_config` writes one JSON line to stdout for each prediction.
7. **Container:** `Dockerfile` builds the image `mp093-api:1.0`.
8. **CI:** `.github/workflows/ci.yml` (GitHub Actions) runs `pytest` on every push and pull request.
9. **Kubernetes:** `k8s/deployment.yaml` (2 replicas, health probes) and `k8s/service.yaml`.
10. **Monitoring:** `mp093.psi_cleaned` computes PSI per numeric column; `mp093.monitoring_analysis` computes v1's MAE per block.
11. **Retraining:** `mp093.monitoring_analysis` retrains v2; `mp093.register_v2` promotes it only if its MAE is at least 5% lower (v2 is now `@champion`).
12. **Final predictions:** `mp093.holdout_predictions` writes `predictions/holdout_predictions.csv`.

## Results
All MAE values are in kg (lower is better). Sources: `reports/findings.yaml`
and `mlflow.db`.

| Measure | Value |
|---|---|
| v1 (ridge) validation MAE (`mlflow.db`) | 55.07 |
| Most drifted feature, cleaned `train.csv` vs `later.csv` | `working_hours`, PSI 5.3029 |
| Worst-served block, v1 on `later.csv` | Block East, MAE 595.49 |
| v1 MAE on the last 30% of `later.csv` (by date) | 292.9967 |
| v2 MAE on the same rows | 205.8379 |
| Decision (at least 5% lower MAE required) | promote v2 |

## Repository layout
```
.
├── .dvc/          # DVC configuration
├── .github/       # GitHub Actions CI workflow
├── app/           # FastAPI service
├── data/          # raw data: holdout features and the .dvc pointer files
├── k8s/           # Kubernetes Deployment and Service
├── models/        # saved models: model.joblib (served) and model_v2.joblib
├── predictions/   # holdout predictions
├── reports/       # findings.yaml, data inspection and model comparison
├── src/           # mp093 package and early exploration scripts
└── tests/         # pytest tests
```

## Technologies
- Python
- Pandas
- Scikit-learn
- DVC
- MLflow
- FastAPI
- Docker
- Kubernetes
- GitHub Actions

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .   # makes `import mp093` work
```

`train.csv` and `later.csv` are tracked with DVC, not git (there is no DVC
remote). Copy them **unchanged** from the problem zip into `data/raw/`, then
check that they match the `.dvc` files:

```powershell
.\.venv\Scripts\dvc.exe status    # must say "Data and pipelines are up to date."
```

On a fresh clone the DVC cache is empty, so `dvc status` first reports
"not in cache"; run `.\.venv\Scripts\dvc.exe commit` once to fill it (the
`.dvc` files do not change when the hashes match).

Never open `train.csv` in Excel and save it: its hash would stop matching.

## Run

| Command | What it does |
|---|---|
| `python -m mp093.check_data` | Looks for the seven data problems in `train.csv` (results go in `reports/findings.yaml`) |
| `python -m mp093.data` | Cleans `train.csv` in memory and checks the result; never writes the file |
| `python -m mp093.train` | Trains every model in `params.yaml`, tracks each run in `mlflow.db`, registers the best as version 1 with alias `@champion`, saves `models/model.joblib` |
| `python -m mp093.psi_cleaned` | Stage 4: PSI (10 bins) of every numeric column, cleaned `train.csv` vs `later.csv` as given; prints the most drifted column |
| `python -m mp093.monitoring_analysis` | Stage 4: v1's MAE on `later.csv` per block, retrains v2 on cleaned `train.csv` + the first 70% of `later.csv` (by date), compares v1 and v2 on the last 30%. Already done: re-running it retrains and logs an extra MLflow run |
| `python -m mp093.register_v2` | Stage 4: registers the saved `models/model_v2.joblib` (no retraining) as version 2 and, if it is at least 5% better, moves `@champion` to it. Runs only once |
| `python -m mp093.holdout_predictions` | Predicts `holdout_features.csv` with `models:/mp093-grain-model@champion` and writes `predictions/holdout_predictions.csv` (`record_id,prediction`) |
| `python -m pytest -q` | Runs all tests (no CSV needed) |

Every setting (paths, columns, detection thresholds, cleaning, MLflow names)
is in `params.yaml`.

### API / Docker / Kubernetes

**Run the API locally** (from the repo root, after the Setup above):

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`GET /health` returns `{"status": "ok", ...}`. `POST /predict` takes exactly
the 8 fields below. `working_hours` above 15 or `cards_attached` below 100,
a missing field or an extra field gets status **422**. Each prediction writes
one JSON log line to stdout.

```powershell
Invoke-RestMethod http://localhost:8000/health

$body = @{
  cards_attached = 742; working_hours = 10.5; biometric_failures = 55
  temp_c = 38.5; stock_kg_morning = 1497; days_to_month_end = 5
  shop_area = "tribal"; block = "Block North"
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -Body $body
```

The same request with curl (bash):

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"cards_attached": 742, "working_hours": 10.5, "biometric_failures": 55, "temp_c": 38.5, "stock_kg_morning": 1497, "days_to_month_end": 5, "shop_area": "tribal", "block": "Block North"}'
```

**Docker** (image `mp093-api:1.0`):

```powershell
docker build -t mp093-api:1.0 .
docker run --rm -p 8000:8000 mp093-api:1.0
```

**Kubernetes** (Deployment `mp093-api`, 2 replicas; Service `mp093-svc`,
port 80 to 8000). Build the image on the cluster's Docker first, then:

```powershell
kubectl apply -f k8s/
kubectl get pods -l app=mp093-api
kubectl port-forward svc/mp093-svc 8080:80   # then call http://localhost:8080/health
```

**CI**: `.github/workflows/ci.yml` runs `pytest -q` on every push and pull request.
