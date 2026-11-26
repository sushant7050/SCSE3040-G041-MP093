
# SCSE3040 - MLOps Mini Project

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
An MLOps project focused on ration shop distribution prediction.

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
