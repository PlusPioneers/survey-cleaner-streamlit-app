import streamlit as st
import pandas as pd
import numpy as np
from src.utils import load_dataframe, generate_sample_dataset
from src.schema import auto_detect_schema, display_schema_editor
from src.cleaning import impute_missing, handle_outliers, apply_rules
from src.weights_stats import compute_summaries, get_weight_series
from src.viz import plot_numeric_hist, plot_categorical_bar, plot_categorical_pie
from src.reporting import generate_html_report, WEASYPRINT_AVAILABLE
from src.chat import display_chat_interface
from src.ui import *

# Page configuration
st.set_page_config(
    page_title="Survey Cleaner & Reporter",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-header {
        background: linear-gradient(90deg, #3498db, #2980b9);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">📊 Survey Cleaner & Reporter</div>', unsafe_allow_html=True)
    st.markdown("**Free-to-use AI-assisted survey data cleaning and analysis tool**")
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    step = st.sidebar.radio(
        "Choose a step:",
        [
            "1️⃣ Upload Data",
            "2️⃣ Configure Schema", 
            "3️⃣ Clean Data",
            "4️⃣ Validation Rules",
            "5️⃣ Weights & Statistics",
            "6️⃣ Visualizations",
            "7️⃣ Generate Report",
            "8️⃣ Export Data",
            "9️⃣ Chat with Data"
        ]
    )
    
    # Step 1: Upload Data
    if step == "1️⃣ Upload Data":
        st.markdown('<div class="step-header"><h2>Step 1: Upload Your Survey Data</h2></div>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = display_file_uploader()
        
        if uploaded_file:
            df = load_dataframe(uploaded_file)
            if not df.empty:
                st.session_state.df_raw = df
                st.session_state.df_clean = df.copy()
                st.session_state.data_cleaned = False
                st.success(f"✅ Data loaded successfully! {len(df)} rows, {len(df.columns)} columns")
                display_data_preview(df, "Uploaded Data")
        
        # Sample data option
        sample_df = display_sample_data_option()
        if sample_df is not None:
            st.session_state.df_raw = sample_df
            st.session_state.df_clean = sample_df.copy()
            st.session_state.data_cleaned = False
            st.success("✅ Sample data loaded successfully!")
            display_data_preview(sample_df, "Sample Data")
    
    # Step 2: Configure Schema
    elif step == "2️⃣ Configure Schema":
        st.markdown('<div class="step-header"><h2>Step 2: Configure Data Schema</h2></div>', unsafe_allow_html=True)
        
        if st.session_state.df_raw.empty:
            st.warning("⚠️ Please upload data first!")
            return
        
        # Auto-detect schema
        if not st.session_state.schema:
            st.session_state.schema = auto_detect_schema(st.session_state.df_raw)
        
        # Display schema editor
        schema, weight_col, const_weight = display_schema_editor(
            st.session_state.df_raw, 
            st.session_state.schema
        )
        
        st.session_state.schema = schema
        st.session_state.weight_col = weight_col
        st.session_state.const_weight = const_weight
        
        if st.button("✅ Confirm Schema"):
            st.success("Schema configuration saved!")
            st.balloons()
    
    # Step 3: Clean Data
    elif step == "3️⃣ Clean Data":
        st.markdown('<div class="step-header"><h2>Step 3: Clean Your Data</h2></div>', unsafe_allow_html=True)
        
        if st.session_state.df_raw.empty:
            st.warning("⚠️ Please upload data first!")
            return
        
        if not st.session_state.schema:
            st.warning("⚠️ Please configure schema first!")
            return
        
        # Display cleaning options
        missing_method, outlier_method, outlier_params = display_cleaning_options()
        
        if st.button("🧹 Apply Cleaning"):
            with st.spinner("Cleaning data..."):
                df_clean = st.session_state.df_raw.copy()
                cleaning_log = []
                
                # Handle missing values
                df_clean, missing_log = impute_missing(df_clean, st.session_state.schema, missing_method)
                cleaning_log.extend(missing_log)
                
                # Handle outliers
                df_clean, outlier_log = handle_outliers(
                    df_clean, 
                    st.session_state.schema, 
                    outlier_method,
                    **outlier_params
                )
                cleaning_log.extend(outlier_log)
                
                st.session_state.df_clean = df_clean
                st.session_state.cleaning_log = cleaning_log
                st.session_state.data_cleaned = True
                
                st.success("✅ Data cleaning completed!")
                display_cleaning_summary(cleaning_log)
        
        # Show cleaned data preview
        if st.session_state.data_cleaned:
            st.subheader("🔍 Cleaned Data Preview")
            display_data_preview(st.session_state.df_clean, "Cleaned Data")
    
    # Step 4: Validation Rules
    elif step == "4️⃣ Validation Rules":
        st.markdown('<div class="step-header"><h2>Step 4: Apply Validation Rules</h2></div>', unsafe_allow_html=True)
        
        if st.session_state.df_clean.empty:
            st.warning("⚠️ Please clean data first!")
            return
        
        # Rule builder
        rules = display_rule_builder(st.session_state.df_clean)
        
        if rules and st.button("⚖️ Apply Rules"):
            with st.spinner("Applying validation rules..."):
                df_validated, violations, rule_log = apply_rules(st.session_state.df_clean, rules)
                
                st.session_state.df_clean = df_validated
                st.session_state.cleaning_log.extend(rule_log)
                
                st.success("✅ Validation rules applied!")
                
                # Show violations if any
                if violations.any():
                    st.warning(f"⚠️ Found {violations.sum()} rule violations")
                    if st.checkbox("Show violating rows"):
                        violating_rows = st.session_state.df_raw.loc[violations]
                        st.dataframe(violating_rows)
    
    # Step 5: Weights & Statistics  
    elif step == "5️⃣ Weights & Statistics":
        st.markdown('<div class="step-header"><h2>Step 5: Compute Weighted Statistics</h2></div>', unsafe_allow_html=True)
        
        if st.session_state.df_clean.empty:
            st.warning("⚠️ Please clean data first!")
            return
        
        st.info(f"Using weight column: {st.session_state.weight_col or 'None (constant weight)'}")
        st.info(f"Constant weight value: {st.session_state.const_weight}")
        
        if st.button("📊 Compute Statistics"):
            with st.spinner("Computing statistics..."):
                summaries = compute_summaries(
                    st.session_state.df_clean,
                    st.session_state.schema,
                    st.session_state.weight_col,
                    st.session_state.const_weight
                )
                
                st.session_state.summaries = summaries
                st.session_state.summaries_computed = True
                
                st.success("✅ Statistics computed!")
        
        # Display summaries
        if st.session_state.summaries_computed:
            summaries = st.session_state.summaries
            
            # Numeric summaries
            if summaries["unweighted"]["numeric"]:
                st.subheader("📈 Numeric Variables")
                
                tab1, tab2 = st.tabs(["Unweighted", "Weighted"])
                
                with tab1:
                    data = []
                    for var, stats in summaries["unweighted"]["numeric"].items():
                        mean_val = stats["mean"]
                        ci_low, ci_high = stats["ci"]
                        data.append({
                            "Variable": var,
                            "Mean": f"{mean_val:.4f}" if not pd.isna(mean_val) else "N/A",
                            "95% CI": f"[{ci_low:.4f}, {ci_high:.4f}]" if not (pd.isna(ci_low) or pd.isna(ci_high)) else "N/A"
                        })
                    st.dataframe(pd.DataFrame(data), use_container_width=True)
                
                with tab2:
                    data = []
                    for var, stats in summaries["weighted"]["numeric"].items():
                        mean_val = stats["mean"]
                        ci_low, ci_high = stats["ci"]
                        data.append({
                            "Variable": var,
                            "Weighted Mean": f"{mean_val:.4f}" if not pd.isna(mean_val) else "N/A",
                            "95% CI": f"[{ci_low:.4f}, {ci_high:.4f}]" if not (pd.isna(ci_low) or pd.isna(ci_high)) else "N/A"
                        })
                    st.dataframe(pd.DataFrame(data), use_container_width=True)
            
            # Categorical summaries
            if summaries["unweighted"]["categorical"]:
                st.subheader("📊 Categorical Variables")
                
                for var, cat_data in summaries["unweighted"]["categorical"].items():
                    st.write(f"**{var}**")
                    
                    tab1, tab2 = st.tabs(["Unweighted", "Weighted"])
                    
                    with tab1:
                        data = []
                        for item in cat_data:
                            data.append({
                                "Value": item["value"],
                                "Count": item["count"],
                                "Proportion": f"{item['prop']:.4f}" if not pd.isna(item['prop']) else "N/A",
                                "95% CI": f"[{item['ci'][0]:.4f}, {item['ci'][1]:.4f}]" if not (pd.isna(item['ci']) or pd.isna(item['ci'][1])) else "N/A"
                            })
                        st.dataframe(pd.DataFrame(data), use_container_width=True)
                    
                    with tab2:
                        w_data = summaries["weighted"]["categorical"].get(var, [])
                        if w_data:
                            data = []
                            for item in w_data:
                                data.append({
                                    "Value": item["value"],
                                    "Weighted Proportion": f"{item['w_prop']:.4f}" if not pd.isna(item['w_prop']) else "N/A"
                                })
                            st.dataframe(pd.DataFrame(data), use_container_width=True)
    
    # Step 6: Visualizations
    elif step == "6️⃣ Visualizations":
        st.markdown('<div class="step-header"><h2>Step 6: Data Visualizations</h2></div>', unsafe_allow_html=True)
        
        if st.session_state.df_clean.empty:
            st.warning("⚠️ Please clean data first!")
            return
        
        # Select column to visualize
        col_to_plot = st.selectbox("Select column to visualize", st.session_state.df_clean.columns)
        col_type = st.session_state.schema.get(col_to_plot, "categorical")
        
        # Weight option
        use_weights = st.checkbox("Use weights in visualization") and st.session_state.weight_col
        weights = get_weight_series(
            st.session_state.df_clean, 
            st.session_state.weight_col, 
            st.session_state.const_weight
        ) if use_weights else None
        
        # Generate plots based on column type
        if col_type == "numeric":
            st.subheader(f"📈 {col_to_plot} Distribution")
            fig = plot_numeric_hist(st.session_state.df_clean, col_to_plot)
            st.plotly_chart(fig, use_container_width=True)
        
        else:  # categorical
            st.subheader(f"📊 {col_to_plot} Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Bar Chart**")
                fig_bar = plot_categorical_bar(st.session_state.df_clean, col_to_plot, weights)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                st.write("**Pie Chart**")
                fig_pie = plot_categorical_pie(st.session_state.df_clean, col_to_plot, weights)
                st.plotly_chart(fig_pie, use_container_width=True)
    
    # Step 7: Generate Report
    elif step == "7️⃣ Generate Report":
        st.markdown('<div class="step-header"><h2>Step 7: Generate Analysis Report</h2></div>', unsafe_allow_html=True)
        
        if not st.session_state.summaries_computed:
            st.warning("⚠️ Please compute statistics first!")
            return
        
        if st.button("📄 Generate Report"):
            with st.spinner("Generating report..."):
                # Prepare dataset summary
                dataset_summary = {
                    "rows": len(st.session_state.df_clean),
                    "cols": len(st.session_state.df_clean.columns),
                    "types": {
                        "numeric": [col for col, typ in st.session_state.schema.items() if typ == "numeric"],
                        "categorical": [col for col, typ in st.session_state.schema.items() if typ == "categorical"]
                    }
                }
                
                # Sample rows HTML
                sample_rows_html = st.session_state.df_clean.head(10).to_html(
                    classes="table table-striped", 
                    table_id="sample-data",
                    escape=False
                )
                
                # Generate HTML report
                html_report = generate_html_report(
                    dataset_summary=dataset_summary,
                    cleaning_log=st.session_state.cleaning_log,
                    summaries=st.session_state.summaries,
                    schema=st.session_state.schema,
                    sample_rows_html=sample_rows_html
                )
                
                st.session_state.html_report = html_report
                
                st.success("✅ Report generated successfully!")
        
        # Display report options
        if st.session_state.html_report:
            display_report_options(st.session_state.html_report)
            
            # Preview report
            if st.checkbox("📖 Preview Report"):
                st.components.v1.html(st.session_state.html_report, height=600, scrolling=True)
    
    # Step 8: Export Data
    elif step == "8️⃣ Export Data":
        st.markdown('<div class="step-header"><h2>Step 8: Export Cleaned Data</h2></div>', unsafe_allow_html=True)
        
        if st.session_state.df_clean.empty:
            st.warning("⚠️ No cleaned data to export!")
            return
        
        display_export_options(st.session_state.df_clean)
        
        # Show final data summary
        st.subheader("📋 Final Data Summary")
        display_data_preview(st.session_state.df_clean, "Final Cleaned Data")
    
    # Step 9: Chat with Data
    elif step == "9️⃣ Chat with Data":
        st.markdown('<div class="step-header"><h2>Step 9: Chat with Your Data</h2></div>', unsafe_allow_html=True)
        
        if not st.session_state.summaries_computed:
            st.warning("⚠️ Please compute statistics first to enable chat!")
            return
        
        display_chat_interface(
            st.session_state.df_clean,
            st.session_state.schema,
            st.session_state.summaries
        )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.info(
        "📊 **Survey Cleaner & Reporter**\n\n"
        "Free-to-use AI-assisted survey data cleaning and analysis tool.\n\n"
        "Features:\n"
        "• Data cleaning & validation\n"
        "• Weighted statistics\n"
        "• Report generation\n"
        "• AI-powered chat\n\n"
        f"WeasyPrint PDF: {'✅' if WEASYPRINT_AVAILABLE else '❌'}"
    )

if __name__ == "__main__":
    main()
