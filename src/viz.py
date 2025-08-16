import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional

def plot_numeric_hist(df: pd.DataFrame, col: str, title_suffix: str = "") -> go.Figure:
    """Create histogram for numeric column."""
    try:
        fig = px.histogram(
            df, 
            x=col, 
            nbins=30, 
            title=f"Histogram: {col} {title_suffix}".strip(),
            template="plotly_white"
        )
        fig.update_layout(
            xaxis_title=col,
            yaxis_title="Count",
            showlegend=False
        )
        return fig
    except Exception as e:
        # Return empty figure on error
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating histogram: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def plot_categorical_bar(df: pd.DataFrame, col: str, weights: Optional[pd.Series] = None, 
                        title_suffix: str = "") -> go.Figure:
    """Create bar chart for categorical column."""
    try:
        if weights is None:
            counts = df[col].value_counts(dropna=False).reset_index()
            counts.columns = [col, "count"]
            y_col = "count"
            y_title = "Count"
            title_prefix = "Bar Chart"
        else:
            # Weighted counts
            tmp = pd.DataFrame({col: df[col], "weight": weights})
            counts = tmp.groupby(col, dropna=False)["weight"].sum().reset_index()
            counts.columns = [col, "weighted_count"]
            y_col = "weighted_count"
            y_title = "Weighted Count"
            title_prefix = "Weighted Bar Chart"
        
        fig = px.bar(
            counts, 
            x=col, 
            y=y_col, 
            title=f"{title_prefix}: {col} {title_suffix}".strip(),
            template="plotly_white"
        )
        fig.update_layout(
            xaxis_title=col,
            yaxis_title=y_title,
            showlegend=False
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating bar chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def plot_categorical_pie(df: pd.DataFrame, col: str, weights: Optional[pd.Series] = None,
                        title_suffix: str = "") -> go.Figure:
    """Create pie chart for categorical column."""
    try:
        if weights is None:
            counts = df[col].value_counts(dropna=False).reset_index()
            counts.columns = [col, "count"]
            values_col = "count"
            title_prefix = "Pie Chart"
        else:
            # Weighted counts
            tmp = pd.DataFrame({col: df[col], "weight": weights})
            counts = tmp.groupby(col, dropna=False)["weight"].sum().reset_index()
            counts.columns = [col, "weighted_count"]
            values_col = "weighted_count"
            title_prefix = "Weighted Pie Chart"
        
        fig = px.pie(
            counts, 
            names=col, 
            values=values_col, 
            title=f"{title_prefix}: {col} {title_suffix}".strip(),
            template="plotly_white"
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating pie chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

def create_summary_table_fig(summaries: dict, schema: dict, summary_type: str = "unweighted") -> go.Figure:
    """Create a summary table visualization."""
    try:
        fig = go.Figure()
        
        # Numeric summaries
        if summaries[summary_type]["numeric"]:
            numeric_data = []
            for var, stats in summaries[summary_type]["numeric"].items():
                mean_val = stats["mean"]
                ci_low, ci_high = stats["ci"]
                
                mean_str = f"{mean_val:.4f}" if not pd.isna(mean_val) else "N/A"
                ci_str = f"[{ci_low:.4f}, {ci_high:.4f}]" if not (pd.isna(ci_low) or pd.isna(ci_high)) else "N/A"
                
                numeric_data.append([var, mean_str, ci_str])
            
            if numeric_data:
                fig.add_trace(go.Table(
                    header=dict(values=["Variable", "Mean", "95% CI"]),
                    cells=dict(values=list(zip(*numeric_data))),
                    domain=dict(x=[0, 1], y=[0.5, 1])
                ))
        
        fig.update_layout(
            title=f"{summary_type.title()} Summary Statistics",
            height=400
        )
        
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating summary table: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
