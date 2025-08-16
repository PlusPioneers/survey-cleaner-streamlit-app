import io
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional

def load_dataframe(uploaded_file: Optional[st.runtime.uploaded_file_manager.UploadedFile]) -> pd.DataFrame:
    """Load DataFrame from uploaded CSV or Excel file."""
    if uploaded_file is None:
        return pd.DataFrame()
    
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload CSV or Excel.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return pd.DataFrame()

def generate_sample_dataset(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate sample survey dataset for demo purposes."""
    rng = np.random.default_rng(seed)
    
    # Generate sample data
    gender = rng.choice(["Male", "Female", "Other"], size=n, p=[0.48, 0.50, 0.02])
    region = rng.choice(["North", "South", "East", "West"], size=n)
    age = np.clip(rng.normal(40, 12, size=n).round(), 18, 85)
    income = np.clip(rng.normal(60000, 20000, size=n), 10000, 200000).round(0)
    satisfaction = np.clip(rng.normal(3.5, 1.0, size=n), 1, 5).round(1)
    vote_intent = rng.choice(["Party A", "Party B", "Undecided"], size=n, p=[0.4, 0.45, 0.15])
    weight = np.clip(rng.normal(1.0, 0.2, size=n), 0.2, 3.0)
    
    # Inject missing values
    mask_missing = rng.random(n) < 0.05
    income[mask_missing] = np.nan
    satisfaction[rng.random(n) < 0.05] = np.nan
    age[rng.random(n) < 0.03] = np.nan
    
    # Inject inconsistencies for rule testing
    inconsistent_idx = np.where((age < 21) & (rng.random(n) < 0.5))[0]
    income[inconsistent_idx] = 180000
    
    df = pd.DataFrame({
        "gender": gender,
        "region": region,
        "age": age,
        "income": income,
        "satisfaction": satisfaction,
        "vote_intent": vote_intent,
        "weight": weight
    })
    
    return df

def save_sample_dataset():
    """Save sample dataset to data/samples/ directory."""
    import os
    os.makedirs("data/samples", exist_ok=True)
    df = generate_sample_dataset()
    df.to_csv("data/samples/demo_survey.csv", index=False)
    return df

def _coerce(series: pd.Series, val):
    """Try to coerce value to series dtype."""
    try:
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(val)
    except Exception:
        pass
    return val
