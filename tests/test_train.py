"""Tests for mp093.train. No CSV, no MLflow: tiny made-up data only."""

import numpy as np
import pandas as pd
import pytest

from mp093.config import load_params
from mp093.train import build_pipeline, time_split

PARAMS = load_params()


def tiny_data(n=30, seed=0):
    """n random rows with every feature, the target and a date; one missing card count."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "record_id": [f"R{i}" for i in range(n)],
        "date": pd.date_range("2026-01-01", periods=n).strftime("%Y-%m-%d"),
        "cards_attached": rng.integers(200, 900, n).astype(float),
        "working_hours": rng.uniform(4, 12, n),
        "biometric_failures": rng.integers(0, 30, n),
        "temp_c": rng.uniform(15, 45, n),
        "stock_kg_morning": rng.uniform(800, 2500, n),
        "days_to_month_end": rng.integers(0, 31, n),
        "shop_area": rng.choice(["urban", "rural", "tribal"], n),
        "block": rng.choice(["Block North", "Block South", "Block East"], n),
    })
    df.loc[0, "cards_attached"] = np.nan     # the imputer must handle this
    df[PARAMS["target"]] = 2 * df["cards_attached"].fillna(500) + rng.normal(0, 20, n)
    return df


# One test per model in params.yaml, so a new model there is tested too.
@pytest.mark.parametrize("model_cfg", PARAMS["train"]["models"], ids=lambda c: c["name"])
def test_pipeline_fits_and_predicts(model_cfg):
    df = tiny_data()
    X, y = df[PARAMS["features"]], df[PARAMS["target"]]

    pipe = build_pipeline(PARAMS, model_cfg)
    pipe.fit(X, y)
    pred = pipe.predict(X)

    assert pred.shape == (len(df),)
    assert np.isfinite(pred).all()           # the missing value did not break it


def test_pipeline_handles_unseen_category():
    df = tiny_data()
    pipe = build_pipeline(PARAMS, PARAMS["train"]["models"][0])
    pipe.fit(df[PARAMS["features"]], df[PARAMS["target"]])

    new = df[PARAMS["features"]].head(1).copy()
    new["shop_area"] = "semi-urban"          # never seen in training
    assert np.isfinite(pipe.predict(new)).all()


def test_time_split_keeps_latest_dates_for_validation():
    df = tiny_data(n=50).sample(frac=1, random_state=1)   # shuffle the rows

    fit, val = time_split(df, PARAMS)

    assert len(val) == round(50 * PARAMS["train"]["val_fraction"])
    assert len(fit) + len(val) == 50
    assert fit["date"].max() <= val["date"].min()        # validation is later in time
