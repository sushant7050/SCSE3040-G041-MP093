from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
from mp093.data import clean, load_raw
import os
import mlflow
import mlflow.sklearn

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
    # Keep the file once it exists: it is the exact object registered as
    # version 2 (mp093.register_v2). A refit differs only by float noise.
    v2_file = ROOT / "models" / "model_v2.joblib"
    if v2_file.exists():
        print(f"{v2_file.relative_to(ROOT)} already exists (registered v2); not overwritten.")
    else:
        joblib.dump(v2_model, v2_file)
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

# Registering v2 and moving @champion is NOT done here: it is done once by
# `python -m mp093.register_v2`, which logs the saved models/model_v2.joblib
# (with data_sha256) as version 2, without retraining.


def load_v1(params):
    """v1 is registry version 1. models/model.joblib now holds the champion (v2)."""
    os.chdir(ROOT)   # so sqlite:///mlflow.db is the repo's database
    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    name = params["mlflow"]["registered_model_name"]
    return mlflow.sklearn.load_model(f"models:/{name}/1")


def main():
    params = load_params()

    model = load_v1(params)

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
    print("\nTo register v2 and move @champion: python -m mp093.register_v2")


if __name__ == "__main__":
    main()