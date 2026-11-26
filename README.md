
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
