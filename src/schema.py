import pandas as pd
import streamlit as st
from typing import Dict

def auto_detect_schema(df: pd.DataFrame, max_unique_for_cat: int = 20) -> Dict[str, str]:
    """Auto-detect column types based on data characteristics."""
    schema = {}
    
    for col in df.columns:
        series = df[col]
        
        if pd.api.types.is_numeric_dtype(series):
            # Numeric unless few unique values suggest categorical codes
            nunique = series.dropna().nunique()
            if nunique <= max_unique_for_cat and nunique > 1:
                schema[col] = "categorical"
            else:
                schema[col] = "numeric"
        else:
            # Object/bool/category assumed categorical
            schema[col] = "categorical"
    
    return schema

def display_schema_editor(df: pd.DataFrame, initial_schema: Dict[str, str]) -> tuple:
    """Display schema editor UI and return updated schema and weight column."""
    st.subheader("📋 Schema Configuration")
    st.info("Review and adjust column types. Select a weight column if available.")
    
    # Create columns for layout
    col1, col2 = st.columns([3, 1])
    
    schema = {}
    
    with col1:
        st.write("**Column Type Mapping:**")
        
        for col in df.columns:
            col_type = st.selectbox(
                f"Column: **{col}**",
                options=["categorical", "numeric"],
                index=0 if initial_schema.get(col, "categorical") == "categorical" else 1,
                key=f"schema_{col}",
                help=f"Current values: {df[col].dropna().unique()[:5].tolist()}"
            )
            schema[col] = col_type
    
    with col2:
        st.write("**Weight Column:**")
        weight_options = ["None"] + [col for col in df.columns if schema.get(col) == "numeric"]
        weight_col = st.selectbox(
            "Select weight column",
            options=weight_options,
            help="Choose a numeric column to use as survey weights"
        )
        
        if weight_col == "None":
            weight_col = None
            const_weight = st.number_input(
                "Constant weight value",
                min_value=0.01,
                value=1.0,
                step=0.1,
                help="Weight to apply to all observations"
            )
        else:
            const_weight = 1.0
    
    return schema, weight_col, const_weight
