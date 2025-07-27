import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

def load_data(path, target_col):
    df = pd.read_csv(path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# -----------------------------
# Logistic Regression - Fraud
# -----------------------------
def test_logistic_regression_fraud():
    X_train, X_test, y_train, y_test = load_data("data/processed/train_ready.csv", "class")
    model = LogisticRegression(max_iter=3000, solver="saga", class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = f1_score(y_test, preds)
    assert score > 0.75, f"Low F1 score for logistic regression on fraud data: {score:.4f}"

# -----------------------------
# LightGBM - Fraud
# -----------------------------
def test_lightgbm_fraud():
    X_train, X_test, y_train, y_test = load_data("data/processed/train_ready.csv", "class")
    model = LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = f1_score(y_test, preds)
    assert score > 0.80, f"Low F1 score for LightGBM on fraud data: {score:.4f}"

# -----------------------------
# Logistic Regression - Creditcard (SMOTE)
# -----------------------------
def test_logistic_regression_credit():
    X_train, X_test, y_train, y_test = load_data("data/processed/creditcard_balanced_smote.csv", "Class")
    model = LogisticRegression(max_iter=3000, solver="saga", class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = f1_score(y_test, preds)
    assert score > 0.65, f"Low F1 score for logistic regression on creditcard data: {score:.4f}"

# -----------------------------
# LightGBM - Creditcard (SMOTE)
# -----------------------------
def test_lightgbm_credit():
    X_train, X_test, y_train, y_test = load_data("data/processed/creditcard_balanced_smote.csv", "Class")
    model = LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = f1_score(y_test, preds)
    assert score > 0.75, f"Low F1 score for LightGBM on creditcard data: {score:.4f}"
