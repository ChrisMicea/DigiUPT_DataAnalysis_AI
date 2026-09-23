# Data Analysis with Python and AI

This repository contains the work completed for the **"Data Analysis with Python and AI"** course in the **2026 DigiUPT program**.

## Repository Structure

```
Data_Analysis/
├── .venv/                    # Shared virtual environment
├── requirements.txt          # Python dependencies
├── Modul4/                   # Titanic Explorer Streamlit App
│   ├── data/
│   │   └── titanic.csv
│   └── main.py
├── Final_Project/            # Student Performance Analysis
│   ├── data/
│   │   └── StudentPerformanceFactors.csv
│   └── Project.ipynb
└── DatAn_Python_AI_Course_2/ # E-bike Analysis & Titanic Data
    ├── data/
    │   ├── ebike_distanta_franare.csv
    │   ├── ebike_date_viteze_mari.csv
    │   ├── ebike_date_viteze_mici.csv
    │   └── titanic.xlsx
    ├── ebikes.ipynb
    └── titanic.ipynb
```

## Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository
2. Create and activate the virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Projects

### 1. Final_Project - Student Performance Analysis

Comprehensive analysis of factors influencing student exam scores using machine learning.

**Analysis Overview:**
- Exploratory data analysis on 6,607 student records
- Identified key factors: hours studied, attendance, previous scores, parental involvement, access to resources, learning disabilities
- Trained Linear Regression and Random Forest models
- Model achieved R² = 0.77 with Linear Regression
- Built prediction system for individual students
- Created suggestion system for grade improvement

**Running the analysis:**
```bash
cd Final_Project
jupyter lab Project.ipynb
# or
jupyter notebook Project.ipynb
```

**Key Findings:**
- Attendance and hours studied are the strongest predictors of exam scores
- Linear Regression outperformed Random Forest for this dataset
- The model can predict exam scores with ~0.45 MAE (mean absolute error)

---

### 2. Modul4 - Titanic Explorer & Model

An interactive Streamlit application for exploring the Titanic dataset and training a logistic regression model.

**Features:**
- Load data from Seaborn's built-in Titanic dataset or upload custom CSV
- Interactive filtering by passenger class, sex, age, embarkation port, etc.
- Exploratory Data Analysis with interactive Plotly charts
- Logistic Regression model training with customizable features
- Model evaluation: accuracy, classification report, confusion matrix, ROC curve

**Running the app:**
```bash
cd Modul4
streamlit run main.py
```

**Key Technologies:**
- Streamlit (web app framework)
- Plotly (interactive visualizations)
- scikit-learn (machine learning)
- Seaborn (data visualization)

---

### 3. DatAn_Python_AI_Course_2

#### A. E-bike Braking Distance Analysis (ebikes.ipynb)

Linear regression analysis of e-bike speed vs. stopping distance.

**Analysis:**
- Simple linear regression on speed vs. braking distance
- Polynomial regression (degree 2) for improved accuracy
- Model achieved R² = 0.99 with polynomial features
- Analysis of residuals for model validation

**Key Results:**
- Linear model: y = 0.4532x - 3.6396
- Polynomial model achieved better fit (R² = 0.9886)

#### B. Titanic Data Manipulation (titanic.ipynb)

Demonstration of pandas data manipulation techniques using Titanic passenger data.

**Techniques covered:**
- Reading Excel files with multiple sheets
- Data concatenation (axis=0, axis=1)
- Different join types (inner, outer, left, right)
- Data merging with suffixes

**Running the notebooks:**
```bash
cd DatAn_Python_AI_Course_2
jupyter lab ebikes.ipynb
jupyter lab titanic.ipynb
# or
jupyter notebook ebikes.ipynb
jupyter notebook titanic.ipynb
```

---

## Dependencies

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `seaborn` - Statistical data visualization
- `scikit-learn` - Machine learning algorithms
- `streamlit` - Web application framework
- `plotly` - Interactive plotting
- `openpyxl` - Excel file support
- `jupyter` - Interactive notebook environment
- `ipykernel` - Jupyter kernel for Python
