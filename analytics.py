"""
analytics.py
────────────
Computes and returns model performance metrics for
the analytics dashboard tab in app.py.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve
)

os.makedirs('plots', exist_ok=True)

DARK_BG   = '#0A1628'
CARD_BG   = '#0F1E36'
BORDER    = '#1E3A5F'
CYAN      = '#00D4FF'
GREEN     = '#00C896'
RED       = '#E63B4A'
AMBER     = '#F59E0B'
MUTED     = '#7A9CC8'
TEXT      = '#E8F0FE'

plt.rcParams.update({
    'figure.facecolor' : DARK_BG,
    'axes.facecolor'   : CARD_BG,
    'axes.edgecolor'   : BORDER,
    'axes.labelcolor'  : MUTED,
    'xtick.color'      : MUTED,
    'ytick.color'      : MUTED,
    'text.color'       : TEXT,
    'grid.color'       : BORDER,
    'grid.alpha'       : 0.5,
    'font.family'      : 'sans-serif',
})


def compute_metrics(model, X_test: pd.DataFrame,
                    y_test: pd.Series) -> dict:
    """Compute full performance metrics for one model."""
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    return {
        'accuracy'  : round(accuracy_score(y_test, preds)   * 100, 1),
        'auc'       : round(roc_auc_score(y_test, proba)     * 100, 1),
        'precision' : round(precision_score(y_test, preds)   * 100, 1),
        'recall'    : round(recall_score(y_test, preds)      * 100, 1),
        'f1'        : round(f1_score(y_test, preds)          * 100, 1),
        'preds'     : preds,
        'proba'     : proba,
        'cm'        : confusion_matrix(y_test, preds)
    }


def plot_roc_comparison(models_dict: dict,
                        X_test: pd.DataFrame,
                        y_test: pd.Series,
                        save_path: str = 'plots/analytics_roc.png') -> str:
    """Plot ROC curves for multiple models on one chart."""
    colors = [CYAN, GREEN, AMBER, RED]
    fig, ax = plt.subplots(figsize=(7, 5))

    for (name, model), color in zip(models_dict.items(), colors):
        proba    = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc      = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{name} (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], '--', color=BORDER, linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate',  fontsize=11)
    ax.set_title('ROC Curve Comparison', fontsize=13, color=TEXT, pad=12)
    ax.legend(fontsize=10, facecolor=CARD_BG, edgecolor=BORDER,
              labelcolor=TEXT)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none')
    plt.close()
    return save_path


def plot_precision_recall(model, X_test: pd.DataFrame,
                          y_test: pd.Series,
                          save_path: str = 'plots/analytics_pr.png') -> str:
    """Plot precision-recall curve."""
    proba      = model.predict_proba(X_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, color=CYAN, linewidth=2)
    ax.fill_between(rec, prec, alpha=0.1, color=CYAN)
    ax.set_xlabel('Recall',    fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curve', fontsize=13,
                 color=TEXT, pad=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none')
    plt.close()
    return save_path


def plot_confusion_matrix(cm: np.ndarray, model_name: str,
                          save_path: str = 'plots/analytics_cm.png') -> str:
    """Plot styled confusion matrix."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues', vmin=0)
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Risk', 'Clot Risk'], fontsize=11)
    ax.set_yticklabels(['No Risk', 'Clot Risk'], fontsize=11)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual',    fontsize=11)
    ax.set_title(f'Confusion Matrix — {model_name}',
                 fontsize=12, color=TEXT, pad=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=16, color=TEXT, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none')
    plt.close()
    return save_path


def plot_feature_importance(model, feature_names: list,
                            save_path: str = 'plots/analytics_feat.png') -> str:
    """Bar chart of top-10 XGBoost feature importances."""
    importances = model.feature_importances_
    feat_df     = pd.Series(importances, index=feature_names)
    feat_df     = feat_df.sort_values(ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(feat_df.index, feat_df.values,
                   color=CYAN, alpha=0.85, height=0.6)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{w:.3f}', va='center', fontsize=9, color=MUTED)
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_title('Top Feature Importances — XGBoost',
                 fontsize=13, color=TEXT, pad=12)
    ax.grid(axis='x', alpha=0.3)
    ax.set_facecolor(CARD_BG)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none')
    plt.close()
    return save_path