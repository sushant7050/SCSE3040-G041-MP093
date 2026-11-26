"""Tests for mp093.data.clean(). No CSV is read: each test builds its own rows."""

import pandas as pd
import pytest

from mp093.config import load_params
from mp093.data import clean


@pytest.fixture
def params():
    return load_params()   # params.yaml is in git, the CSVs are not


def row(record_id, **changes):
    """One normal train.csv row; pass keyword arguments to change columns."""
    r = {
        "record_id": record_id, "date": "2026-01-01",
        "cards_attached": 500, "working_hours": 8.0, "biometric_failures": 3,
        "temp_c": 30.0, "stock_kg_morning": 1600.0, "days_to_month_end": 10,
        "shop_area": "urban", "block": "Block North",
        "kg_entered_in_epos_at_close": 1200.0, "grain_kg": 1210.0,
    }
    r.update(changes)
    return r


def test_clean_removes_duplicates(params):
    raw = pd.DataFrame([row("R1"), row("R1"), row("R2", temp_c=25.0)])

    out = clean(raw, params)

    assert len(out) == 2                       # the extra copy of R1 is gone
    assert sorted(out["record_id"]) == ["R1", "R2"]
    assert len(raw) == 3                       # the input was not changed


def test_clean_fixes_block_south_unit(params):
    unit = params["cleaning"]["unit_fix"]
    raw = pd.DataFrame([
        row("R1", block=unit["group_value"], stock_kg_morning=1600.0 * unit["factor"]),
        row("R2", block="Block North", stock_kg_morning=1600.0),
    ])

    out = clean(raw, params).set_index("record_id")

    assert out.loc["R1", "stock_kg_morning"] == pytest.approx(1600.0)  # divided
    assert out.loc["R2", "stock_kg_morning"] == pytest.approx(1600.0)  # untouched


def test_clean_sentinel_to_missing_and_drops_leak(params):
    s = params["cleaning"]["sentinel"]
    leak = params["cleaning"]["leaking_column"]
    raw = pd.DataFrame([row("R1", **{s["column"]: s["value"]}), row("R2")])

    out = clean(raw, params).set_index("record_id")

    assert pd.isna(out.loc["R1", s["column"]])          # -1 became missing
    assert out.loc["R2", s["column"]] == 500             # a real value is kept
    assert (out[s["column"]] == s["value"]).sum() == 0
    assert leak not in out.columns
    assert leak in raw.columns                           # the input was not changed
