"""Predict data/raw/holdout_features.csv with the @champion model from the
MLflow registry (PROBLEM.md section 8) and save
predictions/holdout_predictions.csv with exactly record_id,prediction.

Run from anywhere (after pip install -e .):
    python -m mp093.holdout_predictions
"""

import os

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient

from mp093.config import ROOT, load_params
from mp093.data import load_raw


def main():
    params = load_params()
    m = params["mlflow"]
    name = m["registered_model_name"]

    os.chdir(ROOT)   # so sqlite:///mlflow.db is the repo's database
    mlflow.set_tracking_uri(m["tracking_uri"])
    version = MlflowClient().get_model_version_by_alias(name, "champion").version
    model = mlflow.sklearn.load_model(f"models:/{name}@champion")

    holdout = load_raw(params, "holdout")
    id_column = params["id_column"]

    output = pd.DataFrame({
        id_column: holdout[id_column],
        "prediction": model.predict(holdout[params["features"]]),
    })

    output_dir = ROOT / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "holdout_predictions.csv", index=False)

    print(f"Model: {name}@champion (version {version})")
    print("Holdout predictions generated successfully.")
    print(f"Rows: {len(output)}")
    print(f"Columns: {list(output.columns)}")
    print(f"Saved to: {output_dir / 'holdout_predictions.csv'}")


if __name__ == "__main__":
    main()
