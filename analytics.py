"""
analytics.py
------------
Clean white-theme charts for the analytics dashboard.
All charts use the same light palette as the app UI.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve
)

os.makedirs('plots', exist_ok=True)

# ── Clean white palette matching app UI ───────────────────────────
BG       = '#ffffff'
PANEL    = '#f8fafc'
BORDER   = '#e2e8f0'
TEAL     = '#0891b2'
TEAL2    = '#0369a1'
GREEN    = '#16a34a'
RED      = '#e11d48'
AMBER    = '#d97706'
PURPLE   = '#9333ea'
TEXT     = '#0f172a'
SUBTEXT  = '#475569'
GRIDCOL  = '#f1f5f9'

# ── Global rcParams — applied once at import ───────────────────────
plt.rcParams.update({
    'figure.facecolor'  : BG,
    'axes.facecolor'    : BG,
    'axes.edgecolor'    : BORDER,
    'axes.labelcolor'   : SUBTEXT,
    'axes.labelsize'    : 11,
    'axes.titlesize'    : 13,
    'axes.titlecolor'   : TEXT,
    'axes.titlepad'     : 14,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : True,
    'grid.color'        : GRIDCOL,
    'grid.linewidth'    : 1,
    'grid.alpha'        : 1.0,
    'xtick.color'       : SUBTEXT,
    'ytick.color'       : SUBTEXT,
    'xtick.labelsize'   : 10,
    'ytick.labelsize'   : 10,
    'text.color'        : TEXT,
    'legend.frameon'    : True,
    'legend.facecolor'  : BG,
    'legend.edgecolor'  : BORDER,
    'legend.fontsize'   : 10,
    'legend.labelcolor' : TEXT,
    'font.family'       : 'sans-serif',
    'figure.dpi'        : 150,
    'savefig.dpi'       : 150,
    'savefig.bbox'      : 'tight',
    'savefig.facecolor' : BG,
    'savefig.edgecolor' : 'none',
    'figure.constrained_layout.use': True,
})

def _style_ax(ax, title='', xlabel='', ylabel=''):
    """Apply consistent clean styling to any axes."""
    ax.set_facecolor(BG)
    ax.tick_params(colors=SUBTEXT, length=3)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(BORDER)
        ax.spines[spine].set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=13, fontweight='600',
                     color=TEXT, pad=14, loc='left')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=SUBTEXT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=SUBTEXT)


def compute_metrics(model, X_test: pd.DataFrame,
                    y_test: pd.Series) -> dict:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        'accuracy'  : round(accuracy_score(y_test, preds)  * 100, 1),
        'auc'       : round(roc_auc_score(y_test, proba)   * 100, 1),
        'precision' : round(precision_score(y_test, preds) * 100, 1),
        'recall'    : round(recall_score(y_test, preds)    * 100, 1),
        'f1'        : round(f1_score(y_test, preds)        * 100, 1),
        'preds'     : preds,
        'proba'     : proba,
        'cm'        : confusion_matrix(y_test, preds),
    }


def plot_roc_comparison(models_dict: dict,
                        X_test: pd.DataFrame,
                        y_test: pd.Series,
                        save_path: str = 'plots/analytics_roc.png') -> str:

    palette = [TEAL, GREEN, AMBER, PURPLE]
    fig, ax = plt.subplots(figsize=(8, 5))

    for (name, model), color in zip(models_dict.items(), palette):
        proba           = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _     = roc_curve(y_test, proba)
        auc             = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f'{name}  (AUC = {auc:.3f})')
        # Shade under curve lightly
        ax.fill_between(fpr, tpr, alpha=0.06, color=color)

    # Random baseline
    ax.plot([0, 1], [0, 1], linestyle='--', color=BORDER,
            linewidth=1.2, label='Random baseline')

    _style_ax(ax,
              title='ROC Curve Comparison',
              xlabel='False Positive Rate',
              ylabel='True Positive Rate')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='lower right', framealpha=1,
              edgecolor=BORDER, fontsize=10)

    # Annotate AUC values on curves
    for (name, model), color in zip(models_dict.items(), palette):
        proba       = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc         = roc_auc_score(y_test, proba)
        mid         = len(fpr) // 3
        ax.annotate(f'AUC={auc:.2f}',
                    xy=(fpr[mid], tpr[mid]),
                    fontsize=8, color=color, fontweight='600',
                    xytext=(6, 0), textcoords='offset points')

    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_precision_recall(model,
                          X_test: pd.DataFrame,
                          y_test: pd.Series,
                          save_path: str = 'plots/analytics_pr.png') -> str:

    proba        = model.predict_proba(X_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, proba)
    auc_pr       = np.trapezoid(prec, rec) if hasattr(np, 'trapezoid') else np.trapz(prec, rec)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rec, prec, color=TEAL, linewidth=2.5,
            label=f'XGBoost  (AP = {abs(auc_pr):.3f})')
    ax.fill_between(rec, prec, alpha=0.08, color=TEAL)

    # Baseline
    baseline = y_test.sum() / len(y_test)
    ax.axhline(baseline, linestyle='--', color=BORDER,
               linewidth=1.2, label=f'Baseline  ({baseline:.2f})')

    _style_ax(ax,
              title='Precision-Recall Curve',
              xlabel='Recall',
              ylabel='Precision')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='upper right', framealpha=1, edgecolor=BORDER)

    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_confusion_matrix(cm: np.ndarray, model_name: str,
                          save_path: str = 'plots/analytics_cm.png') -> str:

    labels = ['No Risk', 'Clot Risk']
    fig, ax = plt.subplots(figsize=(5, 4.5))

    # Draw coloured cells manually — avoids dark imshow backgrounds
    cell_colors = [
        ['#f0fdf4', '#fff1f2'],   # TN=green-tint, FP=red-tint
        ['#fff1f2', '#f0fdf4'],   # FN=red-tint,  TP=green-tint
    ]
    cell_text_colors = [
        [GREEN, RED],
        [RED,   GREEN],
    ]

    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor=cell_colors[i][j],
                edgecolor=BORDER, linewidth=1
            ))
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    fontsize=28, fontweight='700',
                    color=cell_text_colors[i][j])
            ax.text(j, i + 0.3,
                    ['TN','FP','FN','TP'][(i*2)+j],
                    ha='center', va='center',
                    fontsize=9, color=SUBTEXT)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.invert_yaxis()

    _style_ax(ax,
              title=f'Confusion Matrix — {model_name}',
              xlabel='Predicted',
              ylabel='Actual')
    ax.grid(False)

    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_feature_importance(model, feature_names: list,
                            save_path: str = 'plots/analytics_feat.png') -> str:

    importances = model.feature_importances_
    feat_df     = pd.Series(importances, index=feature_names)
    feat_df     = feat_df.sort_values(ascending=True).tail(10)

    # Colour gradient — more important = deeper teal
    n      = len(feat_df)
    colors = [
        plt.cm.Blues(0.35 + 0.55 * (i / max(n - 1, 1)))
        for i in range(n)
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(feat_df.index, feat_df.values,
                   color=colors, height=0.55,
                   edgecolor='none')

    # Value labels
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f'{w:.3f}',
                va='center', fontsize=9,
                color=SUBTEXT, fontweight='500')

    # Vertical reference line at median
    med = feat_df.median()
    ax.axvline(med, color=TEAL, linewidth=1,
               linestyle='--', alpha=0.5,
               label=f'Median ({med:.3f})')

    _style_ax(ax,
              title='Feature Importance — XGBoost (Top 10)',
              xlabel='Importance Score',
              ylabel='')
    ax.set_xlim(0, feat_df.max() * 1.18)
    ax.legend(fontsize=9, framealpha=1, edgecolor=BORDER)
    ax.grid(axis='x', color=GRIDCOL, linewidth=1)
    ax.grid(axis='y', visible=False)

    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_shap_bar(shap_values_mean: pd.Series,
                  save_path: str = 'plots/shap_bar.png') -> str:
    """
    Horizontal bar chart of mean absolute SHAP values.
    Call this with shap_values.abs().mean(axis=0) as input.
    """
    sv = shap_values_mean.sort_values(ascending=True).tail(10)
    pos_color = '#fca5a5'  # light red
    neg_color = '#93c5fd'  # light blue

    colors = [pos_color if v > 0 else neg_color for v in sv.values]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sv.index, sv.values, color=colors,
            height=0.55, edgecolor='none')
    ax.axvline(0, color=BORDER, linewidth=1)

    for i, v in enumerate(sv.values):
        ax.text(v + (0.002 if v >= 0 else -0.002),
                i, f'{v:+.3f}',
                va='center',
                ha='left' if v >= 0 else 'right',
                fontsize=9, color=SUBTEXT)

    _style_ax(ax,
              title='Mean SHAP Values — Feature Impact',
              xlabel='Mean |SHAP value|',
              ylabel='')

    red_p  = mpatches.Patch(color=pos_color, label='Increases risk')
    blue_p = mpatches.Patch(color=neg_color, label='Decreases risk')
    ax.legend(handles=[red_p, blue_p], framealpha=1, edgecolor=BORDER)

    plt.savefig(save_path)
    plt.close()
    return save_path