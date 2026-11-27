from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
from mp093.data import clean, load_raw
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from mp093.config import ROOT, load_params
from mp093.data import load_raw

def retrain_v2(model, params, later):
    date_column = params["date_column"]
    features = params["features"]
    target = params["target"]

    later = later.sort_values(date_column).reset_index(drop=True)

    split_index = int(len(later) * 0.70)

    later_train = later.iloc[:split_index].copy()
    later_test = later.iloc[split_index:].copy()

    original_train = clean(
        load_raw(params, "train"),
        params
    )

    combined_train = pd.concat(
        [original_train, later_train],
        ignore_index=True
    )

    v2_model = clone(model)

    v2_model.fit(
        combined_train[features],
        combined_train[target]
    )
    joblib.dump(v2_model, ROOT / "models" / "model_v2.joblib")
    print("V2 model saved successfully.")

    v1_predictions = model.predict(later_test[features])
    v2_predictions = v2_model.predict(later_test[features])

    v1_mae = mean_absolute_error(
        later_test[target],
        v1_predictions
    )

    v2_mae = mean_absolute_error(
        later_test[target],
        v2_predictions
    )

    improvement = ((v1_mae - v2_mae) / v1_mae) * 100

    print("\nV1 vs V2 RETRAINING EVALUATION")
    print("-" * 40)
    print(f"V1 MAE: {v1_mae:.4f} kg")
    print(f"V2 MAE: {v2_mae:.4f} kg")
    print(f"Improvement: {improvement:.2f}%")

    if improvement >= 5:
        print("Decision: V2 qualifies for promotion.")
    else:
        print("Decision: V2 does not qualify for promotion.")

    return v2_model, v1_mae, v2_mae, improvement

def promote_v2(model, params, improvement):
    if improvement < 5:
        print("\nV2 not promoted: improvement is below 5%.")
        return

    os.chdir(ROOT)

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])

    experiment_name = params["mlflow"]["experiment_name"]
    model_name = params["mlflow"]["registered_model_name"]

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="MP093-v2-retraining") as run:

        mlflow.log_param("model_version", "v2")
        mlflow.log_param("retraining_source", "train.csv + first 70% of later.csv")
        mlflow.log_metric("improvement_percent", improvement)

        info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle",
            registered_model_name=model_name,
        )

    client = MlflowClient()

    version = info.registered_model_version

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=version,
    )

    print("\nV2 PROMOTED SUCCESSFULLY")
    print("Registered model:", model_name)
    print("Version:", version)
    print("Alias: champion")
    print("Run ID:", run.info.run_id)


def main():
    params = load_params()

    model_path = ROOT / "models" / "model.joblib"

    model = joblib.load(model_path)

    later = load_raw(params, "later")

    features = params["features"]
    target = params["target"]
    group_column = params["group_column"]

    predictions = model.predict(later[features])

    later["prediction"] = predictions

    print("\nV1 MAE BY BLOCK")
    print("-" * 40)

    results = []

    for block, group in later.groupby(group_column):
        mae = mean_absolute_error(
            group[target],
            group["prediction"]
        )

        results.append({
            "block": block,
            "mae": mae,
            "rows": len(group),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "mae",
        ascending=False
    )

    print(results_df.to_string(index=False))
    v2_model, v1_mae, v2_mae, improvement = retrain_v2(
        model,
        params,
        later
    )
    promote_v2(
        v2_model,
        params,
        improvement
    )


if __name__ == "__main__":
    main()