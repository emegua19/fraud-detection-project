#  Task 3 – Model Explainability

##  Objective

This task aims to understand *why* our best-performing models predict fraud by applying **SHAP (SHapley Additive exPlanations)**. We focus on both global feature importance (which features generally drive fraud) and local explanations (why a specific transaction was classified as fraud).

---

## 🔍 Tools Used

* `SHAP`: For model-agnostic interpretation
* `LightGBM`: Our best model from Task 2
* `matplotlib`: For visualizing SHAP plots

---

## 🧪 Models Explained

* **LightGBM (Fraud Dataset)** — best-performing model for behavioral and geolocation data.
* **LightGBM (Creditcard Dataset, with SMOTE)** — best-performing model for anonymized transaction data.

---

##  SHAP Summary Plots

### 1. `shap_summary_fraud.png`

This global SHAP plot reveals:

* `amount` and `user_tx_count` are the **top contributors** to fraud detection.
* High transaction frequency (`user_tx_count`, `device_tx_count`) and certain IP-based behaviors strongly influence model outputs.
* Behavioral signals outweigh geolocation features in predictive power.

### 2. `shap_summary_creditcard.png`

* The most influential features are anonymized (e.g., `V14`, `V10`, `V17`).
* Some components strongly drive fraud predictions both positively and negatively — suggesting complex latent patterns.

---

## 🔍 SHAP Force Plots (Local Interpretability)

### 3. `shap_force_fraud_0.png`

* Explains a single fraud prediction.
* Features like **high transaction amount** and **frequent past activity** pushed the prediction toward fraud.
* Geolocation had lower impact.

### 4. `shap_force_creditcard_0.png`

* Shows how latent features (`V14`, `V10`, `V17`) contributed to classifying a specific transaction as fraud.
* The force plot reveals the interplay between opposing factors.

---

##  Key Insights

* **Behavioral features** (transaction amount, user/device frequency) are critical for fraud prediction.
* **SMOTE-balanced models** enable SHAP to identify hidden patterns even in sparse fraud labels.
* The force plots provide valuable **auditability**, explaining individual decisions and increasing trust in the model.

---

## 🗂️ Files Generated

* SHAP summary plots:

  * `reports/figures/shap_fig/shap_summary_fraud.png`
  * `reports/figures/shap_fig/shap_summary_creditcard.png`
* SHAP force plots:

  * `reports/figures/shap_fig/shap_force_fraud_0.png`
  * `reports/figures/shap_fig/shap_force_creditcard_0.png`

