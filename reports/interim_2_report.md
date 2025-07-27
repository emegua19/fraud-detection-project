# 📝 Interim Report - Task 2: Model Building and Training

##  Summary
This report documents the process of building and evaluating fraud detection models using both the **Fraud_Data** and **Creditcard** datasets. The goal was to compare a baseline model (Logistic Regression) against a more powerful ensemble method (LightGBM), and to handle class imbalance issues appropriately using SMOTE.

---

##  1. Data Preparation

###  Datasets Used
- `train_ready.csv`: Cleaned, feature-engineered Fraud_Data (SMOTE applied).
- `creditcard_cleaned_smote.csv`: Cleaned Creditcard data with SMOTE applied.

###  Target Columns
- `class` → for Fraud_Data
- `Class` → for Creditcard

###  Train-Test Split
- Stratified 80-20 split used on both datasets.


---

##  2. Model Selection
Two models were selected:

1. **Logistic Regression**
   - Simple, interpretable baseline.
   - Used `saga` solver with `class_weight='balanced'`.

2. **LightGBM (LGBMClassifier)**
   - Fast and efficient gradient boosting model.
   - Tuned using default hyperparameters for this phase.


---

##  3. Handling Class Imbalance

- **SMOTE** was applied to both datasets to address the significant class imbalance.
- This ensured the model had balanced classes during training.

### Creditcard Data — Class Distribution After SMOTE
- Figure: `reports/figures/creditcard_fig/creditcard_class_dist_after_smote.png`

---

##  4. Model Evaluation

Each model was evaluated using:
- **Confusion Matrix**
- **F1-Score**
- **Precision-Recall Curve**

###  Metrics Used
- **F1-Score**: Good for imbalanced data.
- **PR Curve (AUC-PR)**: Better than ROC in skewed class settings.

---

##  5. Visualizations

### 📁 `reports/figures/creditcard_fig/`
-  `creditcard_class_dist_after_smote.png`

### 📁 `reports/figures/models_fig/`
-  `Logistic_Regression_-_Fraud_confusion_matrix.png`
-  `Logistic_Regression_-_Fraud_pr_curve.png`
-  `LightGBM_-_Fraud_confusion_matrix.png`
-  `LightGBM_-_Fraud_pr_curve.png`
-  `Logistic_Regression_-_Creditcard_(SMOTE)_confusion_matrix.png`
-  `Logistic_Regression_-_Creditcard_(SMOTE)_pr_curve.png`
-  `LightGBM_-_Creditcard_(SMOTE)_confusion_matrix.png`
-  `LightGBM_-_Creditcard_(SMOTE)_pr_curve.png`

---

##  6. Model Comparison

| Dataset     | Model              | F1-Score (Fraud) | AUC-PR / PR Curve | Final Choice       |
|-------------|--------------------|------------------|-------------------|--------------------|
| Fraud_Data  | Logistic Regression | 0.7960           | PR Curve ✔️        | LightGBM ✅         |
| Fraud_Data  | LightGBM            | 0.8209           | PR Curve ✔️        |                    |
| Creditcard  | Logistic Regression | 0.0512 (before)  | Poor               |                    |
| Creditcard  | LightGBM            | 0.5849 (after SMOTE) | Strong improvement | LightGBM ✅         |

### ✅ Conclusion:
- **LightGBM consistently outperforms Logistic Regression** across both datasets.
- After applying SMOTE, **LightGBM on Creditcard data** achieves a significant boost in F1 and PR metrics.

---

## ✅ 7. Code Testing
- Test cases for each model were implemented in `tests/test_models.py`.
- All tests pass after applying SMOTE.

---

## ✅ 8. Next Steps
- Proceed to Task 3: Model Interpretability and SHAP Analysis.
- Evaluate top features contributing to fraud decisions.

---

