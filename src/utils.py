import pandas as pd
import numpy as np

def load_csv(filepath):
    """Load a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(filepath)
        print(f"[INFO] Loaded data: {filepath} | Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load {filepath}: {e}")
        return None

def convert_to_datetime(df, cols):
    """Convert a list of columns to datetime format."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def check_missing_values(df):
    """Print count and percentage of missing values per column."""
    total = df.isnull().sum()
    percent = (total / len(df)) * 100
    return pd.DataFrame({"Missing Count": total, "Missing %": percent}).sort_values("Missing %", ascending=False)
