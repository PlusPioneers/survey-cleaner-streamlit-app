import streamlit as st
import pandas as pd
from typing import Dict, List, Optional
import io

# Tooltips
TOOLTIPS = {
    "uploader": "Drag-and-drop CSV or Excel (.xlsx, .xls) survey data here for analysis.",
    "schema_detect": "Auto-detect column types based on data; adjust mapping if needed.",
    "schema_cat": "Mark variables with a limited set of labels/categories (e.g., gender, region) as categorical.",
    "schema_num": "Mark numeric measures (e.g., age, income, score) as numeric.",
    "schema_weight": "Mark a column to use as survey weights; otherwise define a weight in the UI.",
    "missing": "Choose how to impute missing values: mean, median, mode, or KNN (numeric only for KNN).",
    "outliers": "Outlier handling method for numeric columns: IQR, Z-score, or Winsorization.",
    "rules": "Define rule-based consistency checks (if/then). Rows failing rules can be flagged or dropped.",
    "weights": "Upload a weight column or specify a constant weight; summaries can be weighted or unweighted.",
    "ci": "Compute margins of error and confidence intervals for means and proportions.",
    "report": "Generate standardized HTML/PDF report with dataset summary, cleaning log, and stats tables.",
    "export": "Export cleaned dataset to CSV.",
    "chat": "Ask natural language questions about your processed data using AI."
}

def display_file_uploader():
    """Display file upload interface."""
    st.subheader("📁 Upload Survey Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help=TOOLTIPS["uploader"]
    )
    
    return uploaded_file

def display_sample_data_option():
    """Display option to use sample data."""
    st.subheader("🎯 Or Use Sample Data")
    
    if st.button("Load Sample Survey Data", help="Load a demo dataset with 300 survey responses"):
        from src.utils import generate_sample_dataset
        return generate_sample_dataset()
    
    return None

def display_data_preview(df: pd.DataFrame, title: str = "Data Preview"):
    """Display data preview with basic info."""
    if df.empty:
        return
    
    st.subheader(f"👀 {title}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing %", f"{missing_pct:.1f}")
    
    # Show data types
    st.write("**Column Information:**")
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.astype(str),
        'Missing': df.isnull().sum().values,
        'Unique': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(info_df, use_container_width=True)
    
    # Show sample rows
    st.write("**Sample Rows:**")
    st.dataframe(df.head(10), use_container_width=True)

def display_cleaning_options():
    """Display data cleaning options."""
    st.subheader("🧹 Data Cleaning Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Missing Value Handling:**")
        missing_method = st.selectbox(
            "Imputation method",
            options=["mean", "median", "mode", "knn"],
            help=TOOLTIPS["missing"]
        )
    
    with col2:
        st.write("**Outlier Handling:**")
        outlier_method = st.selectbox(
            "Outlier method",
            options=["none", "iqr", "z-score", "winsorization"],
            help=TOOLTIPS["outliers"]
        )
    
    # Outlier parameters
    outlier_params = {}
    if outlier_method == "iqr":
        outlier_params["iqr_k"] = st.slider("IQR multiplier (k)", 1.0, 3.0, 1.5, 0.1)
    elif outlier_method == "z-score":
        outlier_params["z_thresh"] = st.slider("Z-score threshold", 2.0, 4.0, 3.0, 0.1)
    elif outlier_method == "winsorization":
        lower = st.slider("Lower percentile", 0.0, 0.1, 0.01, 0.01)
        upper = st.slider("Upper percentile", 0.9, 1.0, 0.99, 0.01)
        outlier_params["winsor_limits"] = (lower, upper)
    
    return missing_method, outlier_method, outlier_params

def display_rule_builder(df: pd.DataFrame) -> List[Dict]:
    """Display rule builder interface."""
    st.subheader("⚖️ Rule-Based Validation")
    st.write("Define consistency checks in the form: IF condition THEN condition")
    
    rules = []
    
    if "num_rules" not in st.session_state:
        st.session_state.num_rules = 0
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("➕ Add Rule"):
            st.session_state.num_rules += 1
    with col2:
        if st.button("➖ Remove Rule") and st.session_state.num_rules > 0:
            st.session_state.num_rules -= 1
    
    for i in range(st.session_state.num_rules):
        st.write(f"**Rule {i+1}:**")
        
        col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
        
        with col1:
            if_col = st.selectbox(f"IF column", df.columns, key=f"if_col_{i}")
        with col2:
            if_op = st.selectbox(f"operator", ["==", "!=", "<", "<=", ">", ">=", "in", "not in"], key=f"if_op_{i}")
        with col3:
            if_val = st.text_input(f"value", key=f"if_val_{i}")
        with col4:
            st.write("")  # spacer
        
        col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
        
        with col1:
            then_col = st.selectbox(f"THEN column", df.columns, key=f"then_col_{i}")
        with col2:
            then_op = st.selectbox(f"operator", ["==", "!=", "<", "<=", ">", ">=", "in", "not in"], key=f"then_op_{i}")
        with col3:
            then_val = st.text_input(f"value", key=f"then_val_{i}")
        with col4:
            action = st.selectbox("Action", ["flag", "drop"], key=f"action_{i}")
        
        # Try to parse values
        try:
            if if_op in ["in", "not in"]:
                if_val_parsed = [x.strip() for x in if_val.split(",")]
            else:
                if_val_parsed = pd.to_numeric(if_val) if if_val.replace(".", "").replace("-", "").isdigit() else if_val
            
            if then_op in ["in", "not in"]:
                then_val_parsed = [x.strip() for x in then_val.split(",")]
            else:
                then_val_parsed = pd.to_numeric(then_val) if then_val.replace(".", "").replace("-", "").isdigit() else then_val
            
            rules.append({
                "if_col": if_col,
                "if_op": if_op,
                "if_val": if_val_parsed,
                "then_col": then_col,
                "then_op": then_op,
                "then_val": then_val_parsed,
                "action": action
            })
        except:
            st.error(f"Error parsing rule {i+1}. Check your values.")
    
    return rules

def display_cleaning_summary(cleaning_log: List[str]):
    """Display cleaning operations summary."""
    if cleaning_log:
        st.subheader("📋 Cleaning Summary")
        for log_entry in cleaning_log:
            st.write(f"• {log_entry}")

def display_export_options(df: pd.DataFrame):
    """Display data export options."""
    st.subheader("💾 Export Cleaned Data")
    
    if st.button("📁 Download Cleaned CSV"):
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="cleaned_survey_data.csv",
            mime="text/csv"
        )

def display_report_options(html_report: str):
    """Display report generation options."""
    st.subheader("📊 Generate Report")
    
    if html_report:
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="📄 Download HTML Report",
                data=html_report,
                file_name="survey_analysis_report.html",
                mime="text/html"
            )
        
        with col2:
            try:
                from src.reporting import html_to_pdf, WEASYPRINT_AVAILABLE
                if WEASYPRINT_AVAILABLE:
                    if st.button("📋 Generate PDF Report"):
                        try:
                            pdf_bytes = html_to_pdf(html_report)
                            st.download_button(
                                label="📋 Download PDF Report",
                                data=pdf_bytes,
                                file_name="survey_analysis_report.pdf",
                                mime="application/pdf"
                            )
                        except Exception as e:
                            st.error(f"PDF generation failed: {str(e)}")
                else:
                    st.info("📋 PDF generation not available. WeasyPrint not installed.")
            except ImportError:
                st.info("📋 PDF generation not available. Install WeasyPrint for PDF support.")

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "df_raw": pd.DataFrame(),
        "df_clean": pd.DataFrame(),
        "schema": {},
        "weight_col": None,
        "const_weight": 1.0,
        "cleaning_log": [],
        "summaries": {},
        "html_report": "",
        "data_cleaned": False,
        "summaries_computed": False,
        "chat_history": []
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
