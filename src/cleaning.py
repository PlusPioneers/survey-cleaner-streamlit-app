import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from typing import Dict, List, Tuple
import streamlit as st

def impute_missing(df: pd.DataFrame, schema: Dict[str, str], method: str) -> Tuple[pd.DataFrame, List[str]]:
    """Handle missing values using specified method."""
    log = []
    df_imp = df.copy()
    method = method.lower()
    
    if method not in {"mean", "median", "mode", "knn"}:
        return df_imp, log
    
    # Separate numeric and categorical columns
    num_cols = [c for c, t in schema.items() if t == "numeric" and c in df.columns]
    cat_cols = [c for c, t in schema.items() if t == "categorical" and c in df.columns]
    
    if method in {"mean", "median"}:
        # Handle numeric columns
        for c in num_cols:
            if df_imp[c].isna().any():
                if method == "mean":
                    val = df_imp[c].mean()
                else:
                    val = df_imp[c].median()
                df_imp[c] = df_imp[c].fillna(val)
                log.append(f"Imputed missing in numeric '{c}' with {method}={val:.4f}.")
        
        # Handle categorical columns with mode
        for c in cat_cols:
            if df_imp[c].isna().any():
                mode_val = df_imp[c].mode(dropna=True)
                if len(mode_val) > 0:
                    val = mode_val.iloc[0]
                    df_imp[c] = df_imp[c].fillna(val)
                    log.append(f"Imputed missing in categorical '{c}' with mode='{val}'.")
    
    elif method == "mode":
        # Mode for all columns
        for c in df_imp.columns:
            if df_imp[c].isna().any():
                mode_val = df_imp[c].mode(dropna=True)
                if len(mode_val) > 0:
                    val = mode_val.iloc[0]
                    df_imp[c] = df_imp[c].fillna(val)
                    log.append(f"Imputed missing in '{c}' with mode='{val}'.")
    
    elif method == "knn":
        # KNN for numeric, mode for categorical
        if num_cols:
            try:
                imputer = KNNImputer(n_neighbors=5)
                num_vals = imputer.fit_transform(df_imp[num_cols])
                df_imp[num_cols] = num_vals
                log.append(f"KNN imputation applied to numeric columns: {num_cols}.")
            except Exception as e:
                log.append(f"KNN imputation failed: {e}. Using mean instead.")
                for c in num_cols:
                    if df_imp[c].isna().any():
                        val = df_imp[c].mean()
                        df_imp[c] = df_imp[c].fillna(val)
                        log.append(f"Fallback: Imputed '{c}' with mean={val:.4f}.")
        
        # Mode for categorical
        for c in cat_cols:
            if df_imp[c].isna().any():
                mode_val = df_imp[c].mode(dropna=True)
                if len(mode_val) > 0:
                    val = mode_val.iloc[0]
                    df_imp[c] = df_imp[c].fillna(val)
                    log.append(f"Imputed missing in categorical '{c}' with mode='{val}'.")
    
    return df_imp, log

def handle_outliers(df: pd.DataFrame, schema: Dict[str, str], method: str, 
                   z_thresh: float = 3.0, iqr_k: float = 1.5, 
                   winsor_limits: Tuple[float, float] = (0.01, 0.99)) -> Tuple[pd.DataFrame, List[str]]:
    """Handle outliers in numeric columns."""
    log = []
    df_o = df.copy()
    num_cols = [c for c, t in schema.items() if t == "numeric" and c in df.columns]
    method = method.lower()
    
    if method not in {"iqr", "z-score", "winsorization", "none"}:
        return df_o, log
    
    if method == "none":
        return df_o, log
    
    for c in num_cols:
        s = pd.to_numeric(df_o[c], errors='coerce')
        if s.dropna().empty:
            continue
        
        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - iqr_k * iqr, q3 + iqr_k * iqr
            before = s.copy()
            s = s.clip(lower, upper)
            df_o[c] = s
            count = int((before != s).sum())
            log.append(f"IQR clipping on '{c}' with k={iqr_k}. Adjusted {count} values.")
        
        elif method == "z-score":
            mu, sd = s.mean(), s.std(ddof=0)
            if sd == 0 or np.isnan(sd):
                continue
            z = (s - mu) / sd
            before = s.copy()
            s = s.where(z.abs() <= z_thresh, mu + z_thresh * sd * np.sign(z))
            df_o[c] = s
            count = int((before != s).sum())
            log.append(f"Z-score capping on '{c}' with threshold={z_thresh}. Adjusted {count} values.")
        
        elif method == "winsorization":
            lower_q, upper_q = s.quantile(winsor_limits[0]), s.quantile(winsor_limits[1])
            before = s.copy()
            s = s.clip(lower_q, upper_q)
            df_o[c] = s
            count = int((before != s).sum())
            log.append(f"Winsorization on '{c}' to {winsor_limits}. Adjusted {count} values.")
    
    return df_o, log

def apply_rules(df: pd.DataFrame, rules: List[Dict]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Apply rule-based validation checks."""
    df2 = df.copy()
    violations = pd.Series(False, index=df2.index)
    log = []
    
    for i, r in enumerate(rules):
        try:
            # Apply IF condition
            cond_if = _apply_op(df2[r["if_col"]], r["if_op"], r["if_val"])
            # Apply THEN condition
            cond_then = _apply_op(df2[r["then_col"]], r["then_op"], r["then_val"])
            # Rule violation: IF is true but THEN is false
            rule_viol = cond_if & (~cond_then)
            count = int(rule_viol.sum())
            violations = violations | rule_viol
            
            if r.get("action", "flag") == "drop":
                df2 = df2.loc[~rule_viol].copy()
                violations = violations.loc[df2.index]
                log.append(f"Rule {i+1}: Dropped {count} violating rows.")
            else:
                log.append(f"Rule {i+1}: Flagged {count} violating rows.")
                
        except Exception as e:
            log.append(f"Rule {i+1}: Error applying rule - {e}")
    
    return df2, violations, log

def _apply_op(series: pd.Series, op: str, value) -> pd.Series:
    """Apply comparison operation between series and value."""
    try:
        if op == "==":
            return series == value
        elif op == "!=":
            return series != value
        elif op == "<":
            return series < _coerce_numeric(series, value)
        elif op == "<=":
            return series <= _coerce_numeric(series, value)
        elif op == ">":
            return series > _coerce_numeric(series, value)
        elif op == ">=":
            return series >= _coerce_numeric(series, value)
        elif op == "in":
            if not isinstance(value, list):
                value = [value]
            return series.isin(value)
        elif op == "not in":
            if not isinstance(value, list):
                value = [value]
            return ~series.isin(value)
        elif op == "is null":
            return series.isna()
        elif op == "not null":
            return ~series.isna()
        else:
            raise ValueError(f"Unsupported operator: {op}")
    except Exception:
        return pd.Series(False, index=series.index)

def _coerce_numeric(series: pd.Series, val):
    """Coerce value to numeric if series is numeric."""
    try:
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(val)
    except Exception:
        pass
    return val
