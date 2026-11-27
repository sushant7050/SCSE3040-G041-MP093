"""Register the retrained v2 in MLflow as version 2 and, if the client's rule
says so, move the alias champion to it.

It does NOT retrain: it logs the exact object saved in models/model_v2.joblib
(cleaned train.csv + the first 70% of later.csv by date, see
mp093.monitoring_analysis). v1 is read back from the registry (version 1).

Run once from anywhere (after pip install -e .):
    python -m mp093.register_v2
"""

import hashlib
import os

import joblib
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

from mp093.config import ROOT, data_path, load_params
from mp093.data import clean, load_raw
from mp093.monitoring import should_promote


def later_split(params):
    """later.csv sorted by date; first fraction = v2 training rows, rest = check rows."""
    later = load_raw(params, "later")   # already clean: clean() must NOT be run on it
    later = later.sort_values(params["date_column"], kind="stable").reset_index(drop=True)
    split = int(params["monitoring"]["later_train_fraction"] * len(later))
    return later.iloc[:split], later.iloc[split:], split


def main():
    params = load_params()
    m, mon = params["mlflow"], params["monitoring"]
    name = m["registered_model_name"]
    feats, target = params["features"], params["target"]

    os.chdir(ROOT)   # so sqlite:///mlflow.db is the repo's database
    mlflow.set_tracking_uri(m["tracking_uri"])
    client = MlflowClient()

    try:
        versions = {int(v.version) for v in client.search_model_versions(f"name='{name}'")}
    except MlflowException:
        versions = set()
    if 1 not in versions:
        raise SystemExit(f"'{name}' has no version 1. Run python -m mp093.train first.")
    if 2 in versions:
        raise SystemExit(f"'{name}' already has version 2. Running again would add "
                         "version 3; v2 is registered only once.")

    # The same fingerprint as the v1 runs: the ORIGINAL train.csv, untouched.
    data_sha256 = hashlib.sha256(data_path(params, "train").read_bytes()).hexdigest()
    v1_run = client.get_run(client.get_model_version(name, "1").run_id)
    if v1_run.data.params.get("data_sha256") != data_sha256:
        raise SystemExit("train.csv's sha256 differs from the v1 run's data_sha256.")

    train = clean(load_raw(params, "train"), params)
    later_fit, check, split = later_split(params)

    v1 = mlflow.sklearn.load_model(f"models:/{name}/1")
    v2 = joblib.load(ROOT / mon["model_v2_file"])
    mae_v1 = mean_absolute_error(check[target], v1.predict(check[feats]))
    mae_v2 = mean_absolute_error(check[target], v2.predict(check[feats]))
    promote = should_promote(mae_v1, mae_v2, mon["promote_max_ratio"])
    print(f"check rows {len(check)} (later.csv rows {split}..{split + len(check) - 1} by date)")
    print(f"mae_v1_on_check {mae_v1:.4f}   mae_v2_on_check {mae_v2:.4f}   "
          f"limit {mon['promote_max_ratio']} x v1 = {mon['promote_max_ratio'] * mae_v1:.4f}")
    print(f"decision: {'promote' if promote else 'keep'}")

    mlflow.set_experiment(m["experiment_name"])
    with mlflow.start_run(run_name="v2") as run:
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_params({
            "model_type": type(v2.steps[-1][1]).__name__,
            "train_data": "cleaned train.csv + first "
                          f"{mon['later_train_fraction']:.0%} of later.csv (sorted by date)",
            "n_train_clean": len(train),
            "n_later_train": len(later_fit),
            "n_train_total": len(train) + len(later_fit),
            "later_split_index": split,
            "n_check": len(check),
            "later_train_fraction": mon["later_train_fraction"],
            "promote_max_ratio": mon["promote_max_ratio"],
            "model_file": mon["model_v2_file"],
        })
        mlflow.log_metrics({"mae_v1_on_check": mae_v1, "mae_v2_on_check": mae_v2})
        mlflow.set_tag("role", "v2")
        info = mlflow.sklearn.log_model(
            v2,
            name="model",
            input_example=later_fit[feats].head(3),
            registered_model_name=name,
            # MLflow 3.15+ refuses numpy.dtype by default (handbook section 8)
            serialization_format="cloudpickle",
        )
    version = info.registered_model_version
    print(f"Run v2 {run.info.run_id}: registered {name} version {version}")

    if promote:
        client.set_registered_model_alias(name, "champion", version)
        print(f"Alias @champion -> version {version}")
    else:
        print("Alias @champion stays on version 1 (v2 is not 5% better).")


if __name__ == "__main__":
    main()
