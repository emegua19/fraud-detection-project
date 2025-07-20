# Interim-1 Report: Fraud Detection Project

## 🔍 Objective
Brief summary of the project's goal and Task 1 scope.

## 1. Data Sources
- `Fraud_Data.csv`
- `IpAddress_to_Country.csv`
- `creditcard.csv`

## 2. Missing Value Handling
- Summary of missing columns
- Strategy (e.g., median fill, unknown category)

## 3. Data Cleaning
- Removed duplicates?
- Corrected data types

## 4. Exploratory Data Analysis
- Univariate + bivariate plots summary
- Fraud class insights
- Screenshot references: `report/figures/`

## 5. Feature Engineering
- `hour_of_day`, `day_of_week`
- `time_since_signup`
- `user_tx_count`, etc.

## 6. Geolocation Mapping
- IP address → integer
- Merged with `IpAddress_to_Country.csv`

## 7. Class Imbalance Handling
- Observed ratio: ~9.36%
- SMOTE applied to numeric features
- Justification for technique

## 8. Final Data Preparation
- Scaled numeric features
- One-Hot Encoded categoricals
- Saved to: `train_model_ready.csv`

## Summary
- Task 1 complete
- Ready for Task 2: Modeling
