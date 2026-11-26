"""Load and clean train.csv. Every setting comes from params.yaml.

clean() returns a NEW DataFrame. Nothing here ever writes a CSV file.

Run from anywhere (after pip install -e .) to see the cleaning checked:
    python -m mp093.data
"""

import hashlib

import numpy as np
import pandas as pd

from mp093.config import data_path, load_params


def load_raw(params, name="train"):
    """Read one raw CSV named in params['data'] (train, later or holdout)."""
    return pd.read_csv(data_path(params, name))


def clean(df, params):
    """Fix the problems recorded in reports/findings.yaml.

    Order matters: duplicates are removed first, while every row is still
    exactly as it was typed.
    """
    cfg = params["cleaning"]
    out = df.copy()

    # 1. Drop the extra copies of repeated records.
    if cfg["drop_duplicates"]:
        out = out.drop_duplicates().reset_index(drop=True)

    # 2. One block recorded stock in grams: divide that block's rows only.
    unit = cfg["unit_fix"]
    rows = out[params["group_column"]] == unit["group_value"]
    out[unit["column"]] = out[unit["column"]].astype(float)
    out.loc[rows, unit["column"]] = out.loc[rows, unit["column"]] / unit["factor"]

    # 3. The sentinel number means "blank": make it a real missing value.
    s = cfg["sentinel"]
    out[s["column"]] = out[s["column"]].replace(s["value"], np.nan)

    # 4. Drop the column that is only known after the day is over.
    out = out.drop(columns=[cfg["leaking_column"]])

    return out


def main():
    params = load_params()
    cfg = params["cleaning"]
    train_file = data_path(params, "train")
    sha_before = hashlib.sha256(train_file.read_bytes()).hexdigest()

    raw = load_raw(params, "train")
    cleaned = clean(raw, params)

    print(f"Rows before: {len(raw)}")
    print(f"Rows after:  {len(cleaned)}  ({len(raw) - len(cleaned)} duplicate copies removed)")
    print(f"Columns: {raw.shape[1]} -> {cleaned.shape[1]}  (dropped {cfg['leaking_column']})")

    # Check: no sentinel left, now counted as missing instead.
    s = cfg["sentinel"]
    n_sentinel = int((cleaned[s["column"]] == s["value"]).sum())
    print(f"\n{s['column']} == {s['value']}: {int((raw[s['column']] == s['value']).sum())} rows before, "
          f"{n_sentinel} after; missing now: {int(cleaned[s['column']].isna().sum())}")
    assert n_sentinel == 0, "sentinel value still present"

    # Check: the fixed block's median now sits with the other blocks.
    u = cfg["unit_fix"]
    group = params["group_column"]
    med_before = raw.groupby(group)[u["column"]].median()
    med_after = cleaned.groupby(group)[u["column"]].median()
    print(f"\n{u['column']} median per {group}:")
    print(pd.DataFrame({"before": med_before, "after": med_after}).round(1).to_string())
    others = med_after.drop(u["group_value"]).median()
    ratio = med_after[u["group_value"]] / others
    print(f"{u['group_value']} / other blocks = {ratio:.2f}")
    assert 0.5 < ratio < 2, f"{u['group_value']} still out of line"

    assert cfg["leaking_column"] not in cleaned.columns
    assert cleaned.duplicated().sum() == 0
    assert hashlib.sha256(train_file.read_bytes()).hexdigest() == sha_before, "train.csv changed!"
    print("\nAll checks passed. train.csv unchanged (sha256 identical).")


if __name__ == "__main__":
    main()
