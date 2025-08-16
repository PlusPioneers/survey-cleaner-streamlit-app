import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.proportion import proportion_confint

def get_weight_series(df: pd.DataFrame, weight_col: Optional[str], const_weight: float = 1.0) -> pd.Series:
    """Get weight series from column or constant value."""
    if weight_col and weight_col in df.columns:
        w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0)
        w = w.clip(lower=0)
        return w
    else:
        return pd.Series(np.full(len(df), const_weight), index=df.index)

def weighted_mean_ci(series: pd.Series, weights: pd.Series, alpha=0.05) -> Tuple[float, float, float]:
    """Compute weighted mean with confidence interval using statsmodels."""
    try:
        s = pd.to_numeric(series, errors="coerce")
        mask = s.notna() & weights.notna() & (weights > 0)
        
        if mask.sum() == 0:
            return np.nan, np.nan, np.nan
            
        ds = DescrStatsW(s[mask], weights=weights[mask], ddof=1)
        mean = ds.mean
        lower, upper = ds.zconfint_mean(alpha=alpha)
        return mean, lower, upper
    except Exception:
        return np.nan, np.nan, np.nan

def unweighted_mean_ci(series: pd.Series, alpha=0.05) -> Tuple[float, float, float]:
    """Compute unweighted mean with confidence interval."""
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return np.nan, np.nan, np.nan
        
        mean = s.mean()
        se = s.std(ddof=1) / np.sqrt(len(s))
        z = 1.959963984540054  # ~norm.ppf(0.975)
        lower, upper = mean - z * se, mean + z * se
        return mean, lower, upper
    except Exception:
        return np.nan, np.nan, np.nan

def proportion_ci(count: int, nobs: int, alpha=0.05, method="wilson") -> Tuple[float, float, float]:
    """Compute proportion with confidence interval."""
    try:
        if nobs <= 0:
            return np.nan, np.nan, np.nan
        
        p = count / nobs
        low, up = proportion_confint(count, nobs, alpha=alpha, method=method)
        return p, low, up
    except Exception:
        return np.nan, np.nan, np.nan

def weighted_proportion(series: pd.Series, weights: pd.Series, value) -> float:
    """Compute weighted proportion for a specific value."""
    try:
        mask = series.notna() & weights.notna() & (weights > 0)
        denom = weights[mask].sum()
        if denom == 0:
            return np.nan
        num = weights[mask & (series == value)].sum()
        return num / denom
    except Exception:
        return np.nan

def compute_summaries(df: pd.DataFrame, schema: Dict[str, str], weight_col: Optional[str], 
                     const_weight: float, alpha: float = 0.05) -> Dict:
    """Compute comprehensive weighted and unweighted summaries."""
    results = {
        "unweighted": {"numeric": {}, "categorical": {}},
        "weighted": {"numeric": {}, "categorical": {}}
    }
    
    w = get_weight_series(df, weight_col, const_weight)
    
    # Numeric statistics
    for c, t in schema.items():
        if t != "numeric" or c not in df.columns:
            continue
        
        # Unweighted
        m, lo, hi = unweighted_mean_ci(df[c], alpha=alpha)
        results["unweighted"]["numeric"][c] = {"mean": m, "ci": (lo, hi)}
        
        # Weighted
        mw, low, up = weighted_mean_ci(df[c], w, alpha=alpha)
        results["weighted"]["numeric"][c] = {"mean": mw, "ci": (low, up)}
    
    # Categorical statistics
    for c, t in schema.items():
        if t != "categorical" or c not in df.columns:
            continue
        
        # Unweighted
        vc = df[c].value_counts(dropna=False)
        total = int(df[c].notna().sum())
        cat_info = []
        
        for val, cnt in vc.items():
            p, low, up = proportion_ci(int(cnt if pd.notna(val) else 0), total, alpha=alpha, method="wilson")
            cat_info.append({
                "value": str(val),
                "count": int(cnt),
                "nobs": total,
                "prop": p,
                "ci": (low, up)
            })
        
        results["unweighted"]["categorical"][c] = cat_info
        
        # Weighted proportions
        w_props = []
        unique_vals = df[c].dropna().unique().tolist()
        for val in unique_vals:
            p_w = weighted_proportion(df[c], w, val)
            w_props.append({"value": str(val), "w_prop": p_w})
        
        results["weighted"]["categorical"][c] = w_props
    
    return results
