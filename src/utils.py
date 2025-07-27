import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
)

def load_csv(filepath):
    """Load a CSV file into a DataFrame."""
    try:
        df = pd.read_csv(filepath)
        print(f"[INFO] Loaded data: {filepath} | Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load {filepath}: {e}")
        return None

def convert_to_datetime(df, cols):
    """Convert a list of columns to datetime format."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def check_missing_values(df):
    """Print count and percentage of missing values per column."""
    total = df.isnull().sum()
    percent = (total / len(df)) * 100
    return pd.DataFrame({"Missing Count": total, "Missing %": percent}).sort_values("Missing %", ascending=False)

def evaluate_model(model, X_test, y_test, title="Model", save_path=None):
    """Evaluate model with confusion matrix and PR curve. Optionally save figures."""
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    print(f"\n Evaluation: {title}")
    print("-" * 50)
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(f"{title} - Confusion Matrix")
    plt.grid(False)
    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig_path = os.path.join(save_path, f"{title.replace(' ', '_')}_confusion_matrix.png")
        plt.savefig(fig_path)
        print(f"[Saved] {fig_path}")
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = average_precision_score(y_test, y_scores)

    plt.figure()
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"{title} - PR Curve (AUC={pr_auc:.4f})")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        fig_path = os.path.join(save_path, f"{title.replace(' ', '_')}_pr_curve.png")
        plt.savefig(fig_path)
        print(f"[Saved] {fig_path}")
    plt.show()