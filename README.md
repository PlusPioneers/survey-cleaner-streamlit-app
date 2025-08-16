<<<<<<< HEAD
# Survey Cleaner & Reporter

A free-to-use, AI-assisted web application for survey data cleaning, analysis, and reporting built with Streamlit.

## Features

- **Data Upload**: Drag-and-drop CSV/Excel files
- **Auto Schema Detection**: Automatic column type detection with manual override
- **Data Cleaning**: Missing value imputation, outlier handling, rule-based validation
- **Weighted Statistics**: Compute weighted/unweighted means and confidence intervals
- **Visualizations**: Interactive charts and plots
- **Report Generation**: HTML and PDF reports with statistical summaries
- **AI Chat**: Ask natural language questions about your data using OpenRouter API
- **Export**: Download cleaned datasets as CSV

## Installation (Windows)

### Prerequisites
- Python 3.9 or higher
- Git (optional, for cloning)

### Step 1: Download the Code
Either download the ZIP file or clone with git:
git clone <your-repo-url>
cd survey-cleaner

### Step 2: Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate

### Step 3: Install Dependencies
pip install -r requirements.txt

### Step 4: Install WeasyPrint (Optional for PDF)
For PDF generation, you need to install additional dependencies:

**Option A: Using conda (recommended)**
conda install -c conda-forge weasyprint

**Option B: Manual installation**
1. Download and install GTK+ from: https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer
2. Add GTK to your PATH
3. Install WeasyPrint: `pip install weasyprint`

If PDF generation fails, the app will still work with HTML reports.

### Step 5: Run the Application
streamlit run app.py
=======
# survey-cleaner
>>>>>>> 89adeba0bb17e9db5327ee937d672edda7bfbc4e
