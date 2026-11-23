
import sys
from pathlib import Path

import pandas as pd

# Make src/mp093 importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mp093.cleaning import clean_data


def sample_data():
    return pd.DataFrame({
        "record_id": ["R001", "R002", "R003", "R003"],
        "date": [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
            "2026-01-03"
        ],
        "cards_attached": [500, -1, 600, 600],
        "working_hours": [8.0, 7.5, 9.0, 9.0],
        "biometric_failures": [10, 5, 8, 8],
        "temp_c": [30.0, 32.0, 35.0, 35.0],
        "stock_kg_morning": [1500.0, 2000000.0, 2500.0, 2500.0],
        "days_to_month_end": [30, 29, 28, 28],
        "shop_area": ["rural", "urban", "tribal", "tribal"],
        "block": ["Block North", "Block South", "Block East", "Block East"],
        "kg_entered_in_epos_at_close": [400.0, 500.0, 600.0, 600.0],
        "grain_kg": [390.0, 490.0, 590.0, 590.0]
    })


def test_duplicate_rows_removed():
    df = sample_data()
    cleaned = clean_data(df)

    assert len(cleaned) == 3
    assert cleaned.duplicated().sum() == 0


def test_sentinel_value_replaced():
    df = sample_data()
    cleaned = clean_data(df)

    assert pd.isna(cleaned.loc[1, "cards_attached"])


def test_south_stock_unit_corrected():
    df = sample_data()
    cleaned = clean_data(df)

    south_stock = cleaned.loc[1, "stock_kg_morning"]

    assert south_stock == 2000.0


def test_other_blocks_stock_unchanged():
    df = sample_data()
    cleaned = clean_data(df)

    north_stock = cleaned.loc[0, "stock_kg_morning"]
    east_stock = cleaned.loc[2, "stock_kg_morning"]

    assert north_stock == 1500.0
    assert east_stock == 2500.0


def test_categories_normalized():
    df = sample_data()
    cleaned = clean_data(df)

    assert cleaned["shop_area"].tolist() == [
        "Rural", "Urban", "Tribal"
    ]

    assert cleaned["block"].tolist() == [
        "Block North", "Block South", "Block East"
    ]


def test_dates_converted():
    df = sample_data()
    cleaned = clean_data(df)

    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])


def test_original_dataframe_not_modified():
    df = sample_data()
    original = df.copy(deep=True)

    clean_data(df)

    pd.testing.assert_frame_equal(df, original)


def test_target_column_preserved():
    df = sample_data()
    cleaned = clean_data(df)

    assert "grain_kg" in cleaned.columns
    assert cleaned["grain_kg"].notna().all()
