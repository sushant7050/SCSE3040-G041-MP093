"""Train v1: try every model in params.yaml, track each run in MLflow,
register the best one (by mae_val) as version 1 with the alias champion.

Run from anywhere (after pip install -e .):
    python -m mp093.train
"""

import hashlib
import os

import joblib
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn import ensemble, linear_model
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from mp093.config import ROOT, data_path, load_params
from mp093.data import clean, load_raw

# The "type" names params.yaml may use, and the class each one means.
REGRESSORS = {
    "Ridge": linear_model.Ridge,
    "RandomForestRegressor": ensemble.RandomForestRegressor,
    "GradientBoostingRegressor": ensemble.GradientBoostingRegressor,
}
MODEL_FILE = ROOT / "models" / "model.joblib"


def model_settings(params, model_cfg):
    """The regressor's settings from params.yaml, plus random_state if it takes one."""
    cls = REGRESSORS[model_cfg["type"]]
    settings = dict(model_cfg["params"])
    if "random_state" in cls().get_params():
        settings["random_state"] = params["train"]["random_state"]
    return settings


def build_pipeline(params, model_cfg):
    """Fill missing values, one-hot encode the text columns, then the regressor."""
    t = params["train"]
    prep = ColumnTransformer(
        [
            ("impute", SimpleImputer(strategy="median"), t["impute_median"]),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             t["one_hot"]),
        ],
        remainder="passthrough",   # the other numeric columns go through unchanged
    )
    regressor = REGRESSORS[model_cfg["type"]](**model_settings(params, model_cfg))
    return Pipeline([("prep", prep), ("model", regressor)])


def time_split(df, params):
    """Sort by date; the last val_fraction of rows is the validation set."""
    df = df.sort_values(params["date_column"], kind="stable").reset_index(drop=True)
    n_val = round(len(df) * params["train"]["val_fraction"])
    return df.iloc[:-n_val], df.iloc[-n_val:]


def already_registered(client, name):
    try:
        client.get_registered_model(name)
        return True
    except MlflowException:
        return False


def main():
    params = load_params()
    m = params["mlflow"]
    feats, target = params["features"], params["target"]

    os.chdir(ROOT)   # so sqlite:///mlflow.db and ./mlruns are in the repo root
    mlflow.set_tracking_uri(m["tracking_uri"])
    client = MlflowClient()
    if already_registered(client, m["registered_model_name"]):
        raise SystemExit(f"'{m['registered_model_name']}' is already registered, so v1 "
                         "exists. Running again would add a version, and version 2 "
                         "is meant for the retrained model.")

    data_sha256 = hashlib.sha256(data_path(params, "train").read_bytes()).hexdigest()
    train = clean(load_raw(params, "train"), params)
    fit, val = time_split(train, params)
    later = load_raw(params, "later")   # already clean: clean() must NOT be run on it
    date = params["date_column"]
    print(f"data_sha256 {data_sha256}")
    print(f"fit {len(fit)} rows ({fit[date].min()} .. {fit[date].max()}), "
          f"val {len(val)} rows ({val[date].min()} .. {val[date].max()}), "
          f"later {len(later)} rows\n")

    mlflow.set_experiment(m["experiment_name"])
    results = []
    for cfg in params["train"]["models"]:
        pipe = build_pipeline(params, cfg)
        pipe.fit(fit[feats], fit[target])
        mae_val = mean_absolute_error(val[target], pipe.predict(val[feats]))
        mae_later = mean_absolute_error(later[target], pipe.predict(later[feats]))

        with mlflow.start_run(run_name=cfg["name"]) as run:
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_type", cfg["type"])
            mlflow.log_params(model_settings(params, cfg))
            mlflow.log_params({"val_fraction": params["train"]["val_fraction"],
                               "n_fit": len(fit), "n_val": len(val)})
            mlflow.log_metrics({"mae_val": mae_val, "mae_later": mae_later})
        results.append({"name": cfg["name"], "run_id": run.info.run_id,
                        "mae_val": mae_val, "mae_later": mae_later, "pipe": pipe})
        print(f"{cfg['name']:12s} mae_val {mae_val:8.2f}   mae_later {mae_later:8.2f}")

    best = min(results, key=lambda r: r["mae_val"])   # mae_val only, never mae_later
    print(f"\nBest by mae_val: {best['name']}  run_id {best['run_id']}")

    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.set_tag("role", "v1")
        info = mlflow.sklearn.log_model(
            best["pipe"],
            name="model",
            input_example=fit[feats].dropna().head(3),
            registered_model_name=m["registered_model_name"],
            # MLflow 3.15+ refuses numpy.dtype by default (handbook section 8)
            serialization_format="cloudpickle",
        )
    version = info.registered_model_version
    client.set_registered_model_alias(m["registered_model_name"], "champion", version)
    print(f"Registered {m['registered_model_name']} version {version}, alias @champion")

    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(best["pipe"], MODEL_FILE)
    print(f"Saved {MODEL_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
