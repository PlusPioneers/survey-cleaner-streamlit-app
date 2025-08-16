import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import streamlit as st
import re
import json
import os

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

def get_openrouter_client(api_key: str) -> Optional[OpenAI]:
    """Create OpenRouter client using OpenAI SDK."""
    if not OPENAI_AVAILABLE:
        st.error("OpenAI package not available. Install it for chat functionality.")
        return None
    
    if not api_key:
        return None
    
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        return client
    except Exception as e:
        st.error(f"Failed to create OpenRouter client: {str(e)}")
        return None

def get_api_key() -> Optional[str]:
    """Get API key from multiple sources in order of preference."""
    # Try environment variable first
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        return env_key
    
    # Try Streamlit secrets (for cloud deployment)
    try:
        secrets_key = st.secrets.get("OPENROUTER_API_KEY")
        if secrets_key:
            return secrets_key
    except:
        pass
    
    return None

def build_system_prompt() -> str:
    """Create system prompt for the AI assistant."""
    return """You are a helpful survey data analyst assistant. You have access to a cleaned survey dataset and its statistical summaries.

IMPORTANT RULES:
1. Answer questions ONLY using the provided dataset information
2. If asked about data not in the provided context, say "This information is not available in the current dataset"
3. Be precise with numbers and cite confidence intervals when available
4. Explain statistical concepts clearly for non-technical users
5. If asked to perform calculations, use only the data provided in the context

You should help users understand their survey data through clear explanations and insights."""

def extract_data_context(df: pd.DataFrame, schema: Dict[str, str], summaries: Dict) -> str:
    """Extract key information from the dataset for the AI context."""
    context_parts = []
    
    # Dataset overview
    context_parts.append(f"DATASET OVERVIEW:")
    context_parts.append(f"- Total rows: {len(df)}")
    context_parts.append(f"- Total columns: {len(df.columns)}")
    context_parts.append(f"- Columns: {', '.join(df.columns.tolist())}")
    
    # Schema information
    numeric_cols = [col for col, typ in schema.items() if typ == "numeric"]
    categorical_cols = [col for col, typ in schema.items() if typ == "categorical"]
    
    context_parts.append(f"\nCOLUMN TYPES:")
    context_parts.append(f"- Numeric: {', '.join(numeric_cols) if numeric_cols else 'None'}")
    context_parts.append(f"- Categorical: {', '.join(categorical_cols) if categorical_cols else 'None'}")
    
    # Summary statistics
    if summaries.get("unweighted", {}).get("numeric"):
        context_parts.append(f"\nNUMERIC STATISTICS (Unweighted):")
        for var, stats in summaries["unweighted"]["numeric"].items():
            mean_val = stats["mean"]
            ci_low, ci_high = stats["ci"]
            if not pd.isna(mean_val):
                context_parts.append(f"- {var}: Mean = {mean_val:.4f}")
                if not (pd.isna(ci_low) or pd.isna(ci_high)):
                    context_parts.append(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    
    if summaries.get("unweighted", {}).get("categorical"):
        context_parts.append(f"\nCATEGORICAL STATISTICS (Unweighted):")
        for var, cat_data in summaries["unweighted"]["categorical"].items():
            context_parts.append(f"- {var}:")
            for item in cat_data:
                prop = item["prop"]
                ci_low, ci_high = item["ci"]
                if not pd.isna(prop):
                    context_parts.append(f"  {item['value']}: {item['count']} ({prop:.3f})")
                    if not (pd.isna(ci_low) or pd.isna(ci_high)):
                        context_parts.append(f"    95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    
    # Sample of actual data
    context_parts.append(f"\nSAMPLE DATA (first 5 rows):")
    sample_df = df.head(5)
    context_parts.append(sample_df.to_string(index=False))
    
    return "\n".join(context_parts)

def analyze_user_query(query: str, df: pd.DataFrame, schema: Dict[str, str]) -> Dict:
    """Analyze user query and extract data if needed."""
    query_lower = query.lower()
    
    # Simple query analysis
    analysis = {
        "needs_computation": False,
        "column_focus": [],
        "operation_type": "general",
        "computed_result": None
    }
    
    # Check if query mentions specific columns
    for col in df.columns:
        if col.lower() in query_lower:
            analysis["column_focus"].append(col)
    
    # Check for specific operations
    if any(word in query_lower for word in ["mean", "average", "avg"]):
        analysis["operation_type"] = "mean"
        analysis["needs_computation"] = True
    elif any(word in query_lower for word in ["count", "frequency", "how many"]):
        analysis["operation_type"] = "count"
        analysis["needs_computation"] = True
    elif any(word in query_lower for word in ["group", "by", "breakdown"]):
        analysis["operation_type"] = "groupby"
        analysis["needs_computation"] = True
    
    # Perform computation if needed
    if analysis["needs_computation"] and analysis["column_focus"]:
        try:
            analysis["computed_result"] = perform_computation(
                df, analysis["column_focus"], analysis["operation_type"]
            )
        except Exception as e:
            analysis["computed_result"] = f"Error in computation: {str(e)}"
    
    return analysis

def perform_computation(df: pd.DataFrame, columns: List[str], operation: str) -> str:
    """Perform requested computation on the data."""
    results = []
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if operation == "mean" and pd.api.types.is_numeric_dtype(df[col]):
            mean_val = df[col].mean()
            results.append(f"{col} mean: {mean_val:.4f}")
            
        elif operation == "count":
            if pd.api.types.is_numeric_dtype(df[col]):
                count = df[col].count()
                results.append(f"{col} count (non-null): {count}")
            else:
                value_counts = df[col].value_counts().head(10)
                results.append(f"{col} value counts:\n{value_counts.to_string()}")
                
        elif operation == "groupby" and len(columns) >= 2:
            # Simple groupby operation
            try:
                if pd.api.types.is_numeric_dtype(df[columns[1]]):
                    grouped = df.groupby(columns)[columns[1]].mean()
                    results.append(f"Mean {columns[1]} by {columns}:\n{grouped.to_string()}")
                else:
                    grouped = df.groupby(columns)[columns[1]].value_counts()
                    results.append(f"{columns[1]} counts by {columns}:\n{grouped.to_string()}")
            except Exception as e:
                results.append(f"Groupby error: {str(e)}")
            break  # Only do one groupby
    
    return "\n".join(results) if results else "No computation performed"

def chat_with_data(client: OpenAI, user_query: str, df: pd.DataFrame, 
                  schema: Dict[str, str], summaries: Dict) -> str:
    """Main chat function that processes user query and returns AI response."""
    
    # Analyze query and compute results if needed
    query_analysis = analyze_user_query(user_query, df, schema)
    
    # Build context
    base_context = extract_data_context(df, schema, summaries)
    
    # Add computed results if available
    full_context = base_context
    if query_analysis["computed_result"]:
        full_context += f"\n\nCOMPUTED RESULTS FOR YOUR QUERY:\n{query_analysis['computed_result']}"
    
    # Prepare messages
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": f"Dataset context:\n{full_context}\n\nUser question: {user_query}"}
    ]
    
    try:
        # Call OpenRouter API
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=messages,
            max_tokens=1000,
            temperature=0.7,
            extra_headers={
                "HTTP-Referer": "https://streamlit.io",
                "X-Title": "Survey Cleaner & Reporter"
            }
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error getting AI response: {str(e)}"

def display_chat_interface(df: pd.DataFrame, schema: Dict[str, str], summaries: Dict):
    """Display the chat interface in Streamlit."""
    st.subheader("💬 Chat with Your Data")
    
    # Try to get API key from environment first
    env_api_key = get_api_key()
    
    if env_api_key:
        st.success("🔑 API Key loaded from environment!")
        api_key = env_api_key
        
        # Option to override with manual input
        if st.checkbox("Use different API key"):
            api_key = st.text_input(
                "OpenRouter API Key",
                type="password",
                help="Enter your OpenRouter API key to override the environment key"
            )
    else:
        st.info("🔑 No API key found in environment. Please enter manually:")
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Enter your OpenRouter API key to enable chat functionality. Get one free at openrouter.ai"
        )
    
    if not api_key:
        st.info("Enter your OpenRouter API key above to start chatting with your data.")
        st.markdown("""
        **How to get a free API key:**
        1. Go to [openrouter.ai](https://openrouter.ai)
        2. Sign up for a free account
        3. Navigate to the API Keys section
        4. Create a new API key
        5. Add it to your `.env` file as `OPENROUTER_API_KEY=your_key_here`
        """)
        
        # Show instructions for setting up environment variable
        with st.expander("💡 How to set up automatic API key loading"):
            st.markdown("""
            **Option 1: Using .env file (recommended for local development)**
            1. Create a `.env` file in your project root
            2. Add this line: `OPENROUTER_API_KEY=your_actual_api_key_here`
            3. Restart the application
            
            **Option 2: System environment variable**
            - Windows CMD: `setx OPENROUTER_API_KEY "your_key_here"`
            - Windows PowerShell: `$env:OPENROUTER_API_KEY = "your_key_here"`
            - Mac/Linux: `export OPENROUTER_API_KEY="your_key_here"`
            
            **Option 3: Streamlit Cloud deployment**
            - In your app settings, add to Secrets: `OPENROUTER_API_KEY = "your_key_here"`
            """)
        return
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Create client
    client = get_openrouter_client(api_key)
    if not client:
        return
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask a question about your survey data...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data..."):
                ai_response = chat_with_data(client, user_input, df, schema, summaries)
                st.write(ai_response)
        
        # Add AI response to history
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
    
    # Clear chat button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Show API usage info
    with st.expander("ℹ️ API Information"):
        st.markdown("""
        **Model Used:** DeepSeek V3 (Free Tier)
        **Rate Limits:** Check OpenRouter documentation
        **Privacy:** Your data is processed locally and only statistical summaries are sent to the API
        
        **Example Questions:**
        - "What's the average income by region?"
        - "Show me the distribution of satisfaction scores"
        - "How many respondents chose Party A?"
        - "What's the correlation between age and income?"
        """)
