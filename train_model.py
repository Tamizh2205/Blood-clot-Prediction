import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from xgboost import XGBClassifier

os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ── Load preprocessed data ─────────────────────────────────────────
X_train = pd.read_csv('data/X_train.csv')
X_test  = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test  = pd.read_csv('data/y_test.csv').squeeze()

print("Data loaded successfully!")
print(f"Train: {X_train.shape} | Test: {X_test.shape}\n")

# ══════════════════════════════════════════════════════════════════
# MODEL 1: Random Forest
# ══════════════════════════════════════════════════════════════════
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

rf_acc = accuracy_score(y_test, rf_preds)
rf_auc = roc_auc_score(y_test, rf_proba)
print(f"Random Forest — Accuracy: {rf_acc:.2f} | AUC: {rf_auc:.2f}")

# ══════════════════════════════════════════════════════════════════
# MODEL 2: XGBoost
# ══════════════════════════════════════════════════════════════════
print("\nTraining XGBoost...")
xgb = XGBClassifier(n_estimators=100, random_state=42,
                    eval_metric='logloss', verbosity=0)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_proba = xgb.predict_proba(X_test)[:, 1]

xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_auc = roc_auc_score(y_test, xgb_proba)
print(f"XGBoost       — Accuracy: {xgb_acc:.2f} | AUC: {xgb_auc:.2f}")

# ── Pick best model ────────────────────────────────────────────────
best_model = xgb if xgb_auc >= rf_auc else rf
best_name  = "XGBoost" if xgb_auc >= rf_auc else "Random Forest"
best_proba = xgb_proba if xgb_auc >= rf_auc else rf_proba
best_preds = xgb_preds if xgb_auc >= rf_auc else rf_preds
print(f"\nBest model: {best_name}")

# ── Save models ────────────────────────────────────────────────────
joblib.dump(rf,  'models/random_forest.pkl')
joblib.dump(xgb, 'models/xgboost.pkl')
print("Models saved to /models/")

# ── Classification Report ──────────────────────────────────────────
print(f"\n=== {best_name} Classification Report ===")
print(classification_report(y_test, best_preds,
                             target_names=['No Risk', 'Clot Risk']))

# ── Plot 1: ROC Curve ──────────────────────────────────────────────
plt.figure(figsize=(7, 5))
for name, proba in [("Random Forest", rf_proba), ("XGBoost", xgb_proba)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Blood Clot Risk Models')
plt.legend()
plt.tight_layout()
plt.savefig('plots/roc_curve.png')
plt.show()
print("ROC curve saved: plots/roc_curve.png")

# ── Plot 2: Confusion Matrix ───────────────────────────────────────
plt.figure(figsize=(5, 4))
cm = confusion_matrix(y_test, best_preds)
sns_labels = ['No Risk', 'Clot Risk']
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks([0, 1], sns_labels)
plt.yticks([0, 1], sns_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix — {best_name}')
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center',
                 va='center', fontsize=14, color='black')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png')
plt.show()
print("Confusion matrix saved: plots/confusion_matrix.png")

# ── Plot 3: SHAP Explainability ────────────────────────────────────
print("\nGenerating SHAP explanations...")
explainer   = shap.Explainer(xgb)
shap_values = explainer(X_test)

# SHAP Summary Plot
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig('plots/shap_summary.png')
plt.show()
print("SHAP summary saved: plots/shap_summary.png")

# SHAP Waterfall for first patient
plt.figure()
shap.plots.waterfall(shap_values[0], show=False)
plt.tight_layout()
plt.savefig('plots/shap_waterfall_patient1.png')
plt.show()
print("SHAP waterfall saved: plots/shap_waterfall_patient1.png")

print("\n========================================")
print("Module 2 complete!")
print(f"Best Model : {best_name}")
print(f"Accuracy   : {max(rf_acc, xgb_acc):.2f}")
print(f"AUC Score  : {max(rf_auc, xgb_auc):.2f}")
print("Ready for Module 3 — Streamlit App!")
print("========================================")
