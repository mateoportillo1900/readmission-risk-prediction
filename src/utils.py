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

    Parameters
    ----------
    filename : str
        Name of the CSV file in the data/ directory.
    na_values : list of str
        Additional strings to treat as missing. UCI data uses '?'.

    Returns
    -------
    pd.DataFrame
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
    Convert the UCI readmission label into a binary 30-day target.

    UCI labels:
        '<30', '>30', 'NO'

    New binary:
        readmitted_30 = 1 if '<30'
        readmitted_30 = 0 otherwise
    """
    df = df.copy()

    if source_col not in df.columns:
        raise KeyError(f"Column '{source_col}' not found in dataset.")

    df[target_col] = (df[source_col] == "<30").astype(int)
    return df


# ---------------------------------------------------
# 3. Drop ID + leaky columns
# ---------------------------------------------------
def drop_id_and_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that are:
    - Identifiers: encounter_id, patient_nbr
    - Commonly removed due to high missingness: weight, payer_code, medical_specialty

    You can modify this function after EDA if needed.
    """
    df = df.copy()

    cols_to_drop = [
        "encounter_id",
        "patient_nbr",
        "weight",
        "payer_code",
        "medical_specialty",
    ]

    existing_cols = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_cols, errors="ignore")
    return df
