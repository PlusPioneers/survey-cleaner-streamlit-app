import pandas as pd
from datetime import datetime
from jinja2 import Environment, Template
from typing import Dict, List
import os

# PDF generation with graceful fallback
WEASYPRINT_AVAILABLE = False
try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except ImportError as e:
    print(f"WeasyPrint not available: {e}")
    HTML = None
except OSError as e:
    print(f"WeasyPrint dependencies missing: {e}")
    HTML = None

def generate_html_report(dataset_summary: Dict, cleaning_log: List[str], 
                        summaries: Dict, schema: Dict[str, str], 
                        sample_rows_html: str) -> str:
    """Generate HTML report using Jinja2 template with safe formatting."""
    
    def safe_float_format(value):
        """Safely format float values, handling tuples and None values."""
        if value is None:
            return "N/A"
        if isinstance(value, (tuple, list)):
            # If it's a tuple/list, try to format the first element
            v = value[0] if len(value) > 0 else None
            if isinstance(v, (float, int)) and not pd.isna(v):
                return f"{v:.4f}"
            else:
                return "N/A"
        try:
            if pd.isna(value):
                return "N/A"
            return f"{float(value):.4f}"
        except:
            return "N/A"
    
    template_str = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Survey Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1, h2, h3 { color: #2c3e50; }
        .header { border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 30px; }
        .section { margin-bottom: 30px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; font-weight: bold; }
        .numeric-table th { background-color: #e8f4fd; }
        .categorical-table th { background-color: #fff2e8; }
        .cleaning-log { background-color: #f9f9f9; padding: 15px; border-left: 4px solid #3498db; }
        .cleaning-log ul { margin: 0; }
        .dataset-info { background-color: #f8f9fa; padding: 15px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Survey Analysis Report</h1>
        <p><strong>Generated:</strong> {{ timestamp }}</p>
    </div>

    <div class="section">
        <h2>📋 Dataset Summary</h2>
        <div class="dataset-info">
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Rows</td><td>{{ dataset_summary.rows }}</td></tr>
                <tr><td>Total Columns</td><td>{{ dataset_summary.cols }}</td></tr>
                <tr><td>Numeric Columns</td><td>{{ dataset_summary.types.numeric|length }}</td></tr>
                <tr><td>Categorical Columns</td><td>{{ dataset_summary.types.categorical|length }}</td></tr>
            </table>
        </div>
        
        <h3>Column Schema</h3>
        <table>
            <thead>
                <tr><th>Column</th><th>Type</th></tr>
            </thead>
            <tbody>
                {% for col, typ in schema.items() %}
                <tr><td>{{ col }}</td><td>{{ typ }}</td></tr>
                {% endfor %}
            </tbody>
        </table>
    </div>

    {% if cleaning_log %}
    <div class="section">
        <h2>🧹 Data Cleaning Log</h2>
        <div class="cleaning-log">
            <ul>
                {% for log_entry in cleaning_log %}
                <li>{{ log_entry }}</li>
                {% endfor %}
            </ul>
        </div>
    </div>
    {% endif %}

    <div class="section">
        <h2>📊 Statistical Summaries</h2>
        
        {% if summaries.unweighted.numeric|length > 0 %}
        <h3>Numeric Variables (Unweighted)</h3>
        <table class="numeric-table">
            <thead>
                <tr><th>Variable</th><th>Mean</th><th>95% CI</th></tr>
            </thead>
            <tbody>
                {% for var, s in summaries.unweighted.numeric.items() %}
                <tr>
                    <td>{{ var }}</td>
                    <td>{{ s.mean|safe_float }}</td>
                    <td>{% if s.ci[0] is not none and s.ci[1] is not none %}[{{ s.ci|safe_float }}, {{ s.ci[1]|safe_float }}]{% else %}N/A{% endif %}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}

        {% if summaries.weighted.numeric|length > 0 %}
        <h3>Numeric Variables (Weighted)</h3>
        <table class="numeric-table">
            <thead>
                <tr><th>Variable</th><th>Weighted Mean</th><th>95% CI</th></tr>
            </thead>
            <tbody>
                {% for var, s in summaries.weighted.numeric.items() %}
                <tr>
                    <td>{{ var }}</td>
                    <td>{{ s.mean|safe_float }}</td>
                    <td>{% if s.ci[0] is not none and s.ci[1] is not none %}[{{ s.ci|safe_float }}, {{ s.ci[1]|safe_float }}]{% else %}N/A{% endif %}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}

        {% for var, cat_data in summaries.unweighted.categorical.items() %}
        <h3>{{ var }} (Unweighted)</h3>
        <table class="categorical-table">
            <thead>
                <tr><th>Value</th><th>Count</th><th>Proportion</th><th>95% CI</th></tr>
            </thead>
            <tbody>
                {% for r in cat_data %}
                <tr>
                    <td>{{ r.value }}</td>
                    <td>{{ r.count }}</td>
                    <td>{{ r.prop|safe_float }}</td>
                    <td>{% if r.ci[0] is not none and r.ci[1] is not none %}[{{ r.ci|safe_float }}, {{ r.ci[1]|safe_float }}]{% else %}N/A{% endif %}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endfor %}

        {% for var, w_data in summaries.weighted.categorical.items() %}
        <h3>{{ var }} (Weighted)</h3>
        <table class="categorical-table">
            <thead>
                <tr><th>Value</th><th>Weighted Proportion</th></tr>
            </thead>
            <tbody>
                {% for r in w_data %}
                <tr>
                    <td>{{ r.value }}</td>
                    <td>{{ r.w_prop|safe_float }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endfor %}
    </div>

    <div class="section">
        <h2>📋 Sample Data</h2>
        {{ sample_rows_html|safe }}
    </div>

    <div class="section">
        <p><em>Report generated by Survey Cleaner & Reporter</em></p>
    </div>
</body>
</html>
    """
    
    # Create Jinja2 environment with custom filter
    env = Environment()
    env.filters['safe_float'] = safe_float_format
    
    template = env.from_string(template_str)
    
    html_content = template.render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        dataset_summary=dataset_summary,
        cleaning_log=cleaning_log,
        summaries=summaries,
        schema=schema,
        sample_rows_html=sample_rows_html
    )
    
    return html_content

def html_to_pdf(html_content: str) -> bytes:
    """Convert HTML to PDF using WeasyPrint."""
    if not WEASYPRINT_AVAILABLE:
        raise ImportError("WeasyPrint is not available. Install it for PDF generation.")
    
    try:
        pdf_bytes = HTML(string=html_content).write_pdf()
        return pdf_bytes
    except Exception as e:
        raise Exception(f"Failed to generate PDF: {str(e)}")

def save_template():
    """Save the HTML template to reports/templates/report.html.j2"""
    os.makedirs("reports/templates", exist_ok=True)
    # Template is embedded in the function above
    pass
