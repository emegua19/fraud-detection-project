import pandas as pd
import pytest
from datetime import datetime

import os
import sys

# ------------------------------------------------------------------ #
# Import project modules (add src to path)
# ------------------------------------------------------------------ #
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from src.utils import convert_to_datetime

# === Fixtures ===

@pytest.fixture
def raw_sample_df():
    return pd.DataFrame({
        "signup_time": ["2024-07-01 10:00:00", "2024/07/02 11:00:00", None],
        "purchase_time": ["2024-07-03 12:00:00", "invalid", "2024-07-04 15:00:00"],
        "purchase_value": [100.5, None, 200.0],
        "user_id": [1, 2, 2]  # duplicated user_id
    })


# === Tests ===

def test_convert_to_datetime(raw_sample_df):
    df = convert_to_datetime(raw_sample_df.copy(), ["signup_time", "purchase_time"])
    
    assert pd.api.types.is_datetime64_any_dtype(df["signup_time"])
    assert pd.api.types.is_datetime64_any_dtype(df["purchase_time"])
    assert df["purchase_time"].isna().sum() >= 1  # invalid string becomes NaT


def test_missing_value_handling(raw_sample_df):
    df = raw_sample_df.dropna()
    assert df.isnull().sum().sum() == 0  # All NaNs dropped
    assert len(df) == 1  # Only one row remains after dropping all NaNs


def test_duplicate_removal(raw_sample_df):
    df = raw_sample_df.drop_duplicates()
    assert df.duplicated().sum() == 0  # No duplicates left
    assert df.shape[0] <= raw_sample_df.shape[0]  # Should not be larger
