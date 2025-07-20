#  Fraud Detection Project

This project is developed as part of the 10 Academy Week 8–9 challenge to detect fraudulent online transactions using a combination of geolocation, behavioral, and temporal features. It involves full data preparation and modeling, with a focus on building a high-quality, reproducible ML pipeline.

--- 

## 🔄 Project Workflow: System Diagram

The overall system follows a modular ML data pipeline, with each step in its own notebook:

```

```
         ┌────────────────────────┐
         │    Raw Data Sources    │
         │ Fraud_Data, Creditcard│
         └──────────┬─────────────┘
                    │
                    ▼
      ┌────────────────────────────┐
      │ 01_missing_value_handling  │◄───┐
      └──────────┬─────────────────┘    │
                 ▼                      │
  ┌──────────────────────────────---┐   │
  │ 02_cleaning_and_type_conversion │   │
  └──────────┬────────────────────--┘   │
             ▼                          │
┌────────────────────────────┐          │
│ 03_eda_fraud / 04_eda_credit│         │
└──────────┬──────────────────┘         │
           ▼                            │
 ┌───────────────────────────┐          │
 │ 05_geolocation_merge      │  ←─────┐ │
 └──────────┬────────────────┘        | │
            ▼                         │ │
 ┌─────────────────────────────┐      │ │
 │ 06_feature_engineering      │──────┘ │
 └──────────┬──────────────────┘        │
            ▼                           │
 ┌──────────────────────────┐           │
 │ 07_class_imbalance (SMOTE)│──────────┘
 └──────────┬───────────────┘
            ▼
 ┌───────────────────────────┐
 │ 08_scaling_encoding       │
 └──────────┬────────────────┘
            ▼
 ┌──────────────────────────────┐
 │ Final Model-Ready Dataset    │
 │ → train_model_ready.csv      │
 └──────────────────────────────┘
```

```
##  Project Structure

```

fraud-detection-project/
│
├── .github/workflows/ci.yml              # GitHub Actions workflow for CI
├── data/
│   ├── raw/                              # Raw input files (Fraud, Credit, IP)
│   └── processed/                        # Cleaned, merged, balanced, encoded data
├── notebooks/                            # Step-by-step Jupyter notebooks for Task 1
├── reports/
│   ├── figures/                          # Plots and visualizations
│   └── interim\_1\_report.md              # Summary report for Task 1
├── src/                                  # Core logic and reusable code
├── tests/                                # Unit tests
├── README.md                             # This file
├── requirements.txt                      # Python dependencies
├── environment.yml                       # Conda environment (optional)
└── .gitignore                            # Ignored files

````

---

##  Project Objective

The goal is to identify potentially fraudulent purchases using:
- User metadata (e.g., age, device, signup time)
- Transaction behavior (amount, frequency)
- Geolocation info derived from IP address
- Time-based behavioral features (hour, day, time since signup)

---

##  Tasks Completed (Task 1)

### 1. **Data Analysis and Preprocessing**
- Missing value imputation
- Data type fixes and duplicate removal
- Exploratory Data Analysis (EDA) for both fraud and creditcard datasets

### 2. **Feature Engineering**
- Time-based features: `hour_of_day`, `day_of_week`, `time_since_signup`
- Frequency-based features: `user_tx_count`, `device_tx_count`, `ip_tx_count`

### 3. **Geolocation Mapping**
- Converted IP addresses to integers
- Mapped to country using `IpAddress_to_Country.csv`

### 4. **Class Imbalance Handling**
- Applied **SMOTE** to balance fraud class

### 5. **Encoding & Scaling**
- One-hot encoding for categorical features
- Standard scaling for numeric features
- Saved as `train_model_ready.csv`

---

##  Testing

Unit tests are included in the `tests/` directory:

```bash
pytest tests/
````

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/emegua19/fraud-detection-project.git
cd fraud-detection-project
```

### 2. Create Environment

**Using pip:**

```bash
pip install -r requirements.txt
```

**Or with conda:**

```bash
conda env create -f environment.yml
conda activate fraud-detection
```

---

##  Reports & Visualizations

Visual EDA plots and correlation heatmaps are saved in:

```
reports/figures/
```

Interim-1 summary report:

```
reports/interim_1_report.md
```

---

##  Next Steps (Task 2 Preview)

* Build baseline models (Logistic Regression, XGBoost)
* Evaluate with AUC, F1-score, confusion matrix
* Interpret model using SHAP values
* Deploy selected model with FastAPI (optional)

---

##  Author

**Yitbarek Geletaw** – Data Science Fellow @ 10 Academy
💻 [GitHub](https://github.com/emegua19/fraud-detection-project)