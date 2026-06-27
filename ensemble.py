"""
ensemble.py
───────────
Ensemble predictor combining Random Forest + XGBoost.
Uses soft voting (average of probabilities) for higher accuracy.
"""

import numpy as np
import pandas as pd
import joblib


def load_ensemble(rf_path='models/random_forest.pkl',
                  xgb_path='models/xgboost.pkl') -> dict:
    """Load both models and return as ensemble dict."""
    return {
        'rf'  : joblib.load(rf_path),
        'xgb' : joblib.load(xgb_path)
    }


def ensemble_predict(input_data: pd.DataFrame, models: dict,
                     weights: dict = None) -> dict:
    """
    Soft-voting ensemble prediction.

    Parameters
    ----------
    input_data : pd.DataFrame — single row of patient features
    models     : dict         — {'rf': model, 'xgb': model}
    weights    : dict         — {'rf': 0.4, 'xgb': 0.6} (default equal)

    Returns
    -------
    dict with keys: prediction, probability, confidence,
                    rf_prob, xgb_prob, risk_level, agreement
    """
    if weights is None:
        weights = {'rf': 0.4, 'xgb': 0.6}

    rf_prob  = models['rf'].predict_proba(input_data)[0][1]
    xgb_prob = models['xgb'].predict_proba(input_data)[0][1]

    # Weighted soft vote
    ensemble_prob = (weights['rf']  * rf_prob +
                     weights['xgb'] * xgb_prob)

    prediction = 1 if ensemble_prob >= 0.5 else 0
    confidence = round(ensemble_prob * 100, 1)

    # Risk level thresholding
    if ensemble_prob >= 0.70:
        risk_level = 'HIGH'
    elif ensemble_prob >= 0.50:
        risk_level = 'MEDIUM'
    elif ensemble_prob >= 0.30:
        risk_level = 'LOW'
    else:
        risk_level = 'VERY LOW'

    # Confidence check — inconclusive zone
    rf_pred  = 1 if rf_prob  >= 0.5 else 0
    xgb_pred = 1 if xgb_prob >= 0.5 else 0
    agreement = rf_pred == xgb_pred

    # Below 70% confidence → flag as inconclusive
    if confidence < 70 and confidence > 35:
        risk_level = 'INCONCLUSIVE'

    return {
        'prediction'   : prediction,
        'probability'  : ensemble_prob,
        'confidence'   : confidence,
        'rf_prob'      : round(rf_prob  * 100, 1),
        'xgb_prob'     : round(xgb_prob * 100, 1),
        'risk_level'   : risk_level,
        'agreement'    : agreement,
        'weights_used' : weights
    }


def get_model_disagreement_note(result: dict) -> str:
    """Return a human-readable note about model agreement."""
    if result['agreement']:
        return "Both models agree on this prediction."
    diff = abs(result['rf_prob'] - result['xgb_prob'])
    if diff > 30:
        return (f"Models strongly disagree: RF={result['rf_prob']}%, "
                f"XGB={result['xgb_prob']}%. Manual review recommended.")
    return (f"Minor disagreement between models: RF={result['rf_prob']}%, "
            f"XGB={result['xgb_prob']}%.")