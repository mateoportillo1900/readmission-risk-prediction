"""
Utility functions for loading and preprocessing the UCI Hospital Readmissions dataset.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

# --- Paths ---
# PROJECT_ROOT is the repo root, assuming this file lives in src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


# ---------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------
def load_raw_data(
    filename: str = "diabetic_data.csv",
    na_values: list[str] | None = None
) -> pd.DataFrame:
    """
    Load the main UCI dataset as a pandas DataFrame.
    """
    if na_values is None:
        na_values = ["?"]

    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")

    df = pd.read_csv(path, na_values=na_values)
    return df


# ---------------------------------------------------
# 2. Create binary target variable
# ---------------------------------------------------
def add_binary_target(
    df: pd.DataFrame,
    source_col: str = "readmitted",
    target_col: str = "readmitted_30"
) -> pd.DataFrame:
    """
    Convert '<30' into binary 1, all others (NO, >30) into 0.
    """
    df = df.copy()
    df[target_col] = (df[source_col] == "<30").astype(int)
    return df


# ---------------------------------------------------
# 3. Drop ID + leaky columns
# ---------------------------------------------------
def drop_id_and_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops identifier columns and sparse columns.
    """
    df = df.copy()

    cols_to_drop = [
        "encounter_id",
        "patient_nbr",
        "weight",
        "payer_code",
        "medical_specialty"
    ]

    existing_cols = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_cols, errors="ignore")
    return df
