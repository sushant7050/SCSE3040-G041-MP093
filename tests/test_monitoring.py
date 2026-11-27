"""Tests for mp093.monitoring: PSI and the client's 5% promotion rule.

No CSV: every sample is generated inside the test.
"""

import numpy as np
import pandas as pd
import pytest

from mp093.config import load_params
from mp093.monitoring import calculate_psi, should_promote


def sample(n=2000, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(10 + shift, 2, n))


def test_psi_of_identical_samples_is_about_zero():
    s = sample()
    assert calculate_psi(s, s.copy(), bins=10) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_when_one_sample_is_shifted():
    ref = sample(seed=0)
    no_shift = calculate_psi(ref, sample(seed=1), bins=10)
    small = calculate_psi(ref, sample(shift=1, seed=1), bins=10)
    big = calculate_psi(ref, sample(shift=4, seed=1), bins=10)

    assert no_shift < 0.1                  # same distribution, new draw: stable
    assert no_shift < small < big
    assert big > 0.25                      # a big shift counts as major drift


@pytest.mark.parametrize("mae_v1, mae_v2, expected", [
    (100, 94, True),     # 6% lower: promote
    (100, 96, False),    # only 4% lower: keep v1
    (100, 95, True),     # exactly 5% lower still counts
])
def test_should_promote_follows_the_5_percent_rule(mae_v1, mae_v2, expected):
    assert should_promote(mae_v1, mae_v2) is expected


def test_should_promote_uses_the_ratio_from_params():
    ratio = load_params()["monitoring"]["promote_max_ratio"]
    assert ratio == 0.95
    assert should_promote(100, 95, ratio) is True
    assert should_promote(100, 95.01, ratio) is False
