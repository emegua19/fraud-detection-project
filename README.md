
#  Fraud Detection Project

This project is developed as part of the **10 Academy Week 8–9 challenge**, focusing on detecting fraudulent online transactions using a combination of behavioral, geolocation, and time-based features. The solution follows a modular ML pipeline, with comprehensive EDA, class imbalance handling, and model evaluation.

---

##  Project Workflow: System Diagram

The system is organized step-by-step with clear dependencies between notebooks:

```

```
          ┌────────────────────────┐
          │    Raw Data Sources    │
          │ Fraud_Data, Creditcard │
          └──────────┬─────────────┘
                     │
                     ▼
   ┌────────────────────────────────────┐
   │ 01_missing_value_handling.ipynb     │
   └──────────┬─────────────────────────┘
              ▼
   ┌────────────────────────────────────────┐
   │ 02_cleaning_and_type_conversion.ipynb  │
   └──────────┬─────────────────────────────┘
              ▼
 ┌─────────────────────────────┐ ┌────────────────────────────┐
 │ 03_eda_fraud.ipynb          │ │ 04_eda_creditcard.ipynb     │
 └──────────┬──────────────────┘ └────────────┬────────────────┘
            ▼                                 ▼
   ┌────────────────────────────────────┐
   │ 05_geolocation_merge.ipynb         │
   └──────────┬─────────────────────────┘
              ▼
   ┌────────────────────────────────────┐
   │ 06_feature_engineering.ipynb       │
   └──────────┬─────────────────────────┘
              ▼
   ┌────────────────────────────────────┐
   │ 07_class_imbalance.ipynb           │
   └──────────┬─────────────────────────┘
              ▼
   ┌────────────────────────────────────┐
   │ 08_scaling_encoding.ipynb          │
   └──────────┬─────────────────────────┘
              ▼
   ┌────────────────────────────────────┐
   │ 10_smote_creditcard.ipynb          │
   └──────────┬─────────────────────────┘
              ▼
   ┌────────────────────────────────────┐
   │ 09_model_training.ipynb            │
   └────────────────────────────────────┘
```

```

---

## 📂 Project Structure

```

fraud-detection-project/
├── .github/workflows/ci.yml            # GitHub Actions for CI
├── data/
│   ├── raw/                            # Raw input CSVs
│   └── processed/                      # Cleaned, merged, SMOTE-applied data
├── notebooks/                          # Jupyter Notebooks by task
├── reports/
│   ├── figures/
│   │   ├── fraud\_fig/                  # EDA plots for fraud data
│   │   ├── creditcard\_fig/            # Class distribution for credit data
│   │   └── models\_fig/                # Model evaluation plots
│   ├── interim\_1\_report.md
│   └── interim\_2\_report.md
├── src/                                # Utility and configuration scripts
├── tests/                              # Unit tests for core components
├── README.md
├── requirements.txt
├── environment.yml
└── .gitignore

````

---

##  Project Objectives

- Build a fraud detection pipeline using two datasets: `Fraud_Data.csv` and `creditcard.csv`
- Engineer features including geolocation and transaction frequency
- Handle severe class imbalance using **SMOTE**
- Build, evaluate, and compare **Logistic Regression** and **LightGBM**
- Justify the best model using precision, recall, F1-score, and PR curves

---

## ✅ Task Highlights

### Task 1: Data Preparation & EDA
- Handled missing values and duplicates
- Cleaned data types and performed initial EDA
- Engineered features: time of transaction, frequency per user/device/IP
- Merged geolocation via IP-to-country mapping
- Balanced data using **SMOTE** and **Random Undersampling**
- Encoded and scaled data for modeling

📁 Output:
- `train_model_ready.csv`
- Visuals in `reports/figures/fraud_fig/`

---

###  Task 2: Model Building & Evaluation

#### Models Trained:
- **Logistic Regression** (baseline)
- **LightGBM** (ensemble model)

#### Metrics Used:
- **F1-Score**
- **Precision/Recall**
- **AUC-PR**
- **Confusion Matrix**

📁 Model Inputs:
- `data/processed/train_model_ready.csv`
- `data/processed/creditcard_balanced_smote.csv`

📁 Evaluation Visuals:
Saved in `reports/figures/models_fig/`  
Includes PR Curves and Confusion Matrices for all models.

---

## 🧪 Testing

Run all unit tests:
```bash
pytest tests/
````

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/emegua19/fraud-detection-project.git
cd fraud-detection-project
```

### 2. Install Dependencies

With pip:

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda env create -f environment.yml
conda activate fraud-detection
```

---

## 📁 Key Outputs Summary

###  Data Files

* `train_model_ready.csv`
* `creditcard_balanced_smote.csv`

###  Figures

* `fraud_fig/` – EDA plots
* `creditcard_fig/` – Post-SMOTE distribution
* `models_fig/` – Confusion matrices + PR curves

### 📄 Reports

* `interim_1_report.md`
* `interim_2_report.md`

---

##  Author

**Yitbarek Geletaw**
Data Science Fellow @ 10 Academy
🔗 [GitHub](https://github.com/emegua19/fraud-detection-project)

