"""
clinical_panel.py
─────────────────
Reusable clinical data collection + prediction component.
Call render_clinical_form() inside any Streamlit tab to show
the clinical input panel and return encoded input data.
Call run_clinical_prediction() to get risk scores.
"""

import streamlit as st
import pandas as pd


# ── Display labels and encoding maps ──────────────────────────────
CHEST_PAIN_OPTIONS  = ["Typical Angina", "Atypical Angina",
                        "Non-Anginal", "Asymptomatic"]
ECG_OPTIONS         = ["Normal", "ST-T Abnormality",
                        "Left Ventricular Hypertrophy"]
SLOPE_OPTIONS       = ["Upsloping", "Flat", "Downsloping"]
THAL_OPTIONS        = ["Normal", "Fixed Defect", "Reversible Defect"]

SYMPTOM_OPTIONS     = [
    "Chest pain / tightness",
    "Shortness of breath",
    "Leg swelling / pain",
    "Rapid heart rate",
    "Dizziness / fainting",
    "Coughing blood",
    "Recent surgery (< 3 months)",
    "Prolonged immobility (> 4 hours)",
    "Pregnancy or recent delivery",
    "Family history of clotting",
]


def render_clinical_form(tab_key: str) -> dict:
    """
    Renders the clinical input form inside the current tab.
    Returns a dict of raw display values + encoded ML features.
    tab_key: unique string per tab to avoid Streamlit key conflicts.
    """

    st.markdown("""
    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-left:4px solid #2563eb;
                border-radius:10px;padding:12px 16px;margin-bottom:20px;font-size:12px;
                color:#1e40af;line-height:1.6">
      <strong>Combined analysis:</strong> Fill in the patient clinical details below.
      Both the scan model and the clinical risk model will run together to produce
      a unified AI assessment with an overall risk score.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown('<div class="s-divider">Demographics &amp; Vitals</div>',
                    unsafe_allow_html=True)
        age      = st.slider("Age (years)",
                             20, 90, 55, key=f"age_{tab_key}")
        sex      = st.selectbox("Sex",
                             ["Male", "Female"], key=f"sex_{tab_key}")
        bmi      = st.slider("BMI",
                             15.0, 50.0, 25.0, step=0.5, key=f"bmi_{tab_key}")
        smoking  = st.selectbox("Smoking status",
                             ["Never", "Former", "Current"], key=f"smoke_{tab_key}")
        diabetes = st.selectbox("Diabetes",
                             ["No", "Yes"], key=f"diab_{tab_key}")

    with c2:
        st.markdown('<div class="s-divider">Lab Results</div>',
                    unsafe_allow_html=True)
        trestbps = st.slider("Blood pressure (mm Hg)",
                             80, 200, 120, key=f"bp_{tab_key}")
        chol     = st.slider("Cholesterol (mg/dl)",
                             100, 600, 200, key=f"chol_{tab_key}")
        thalach  = st.slider("Max heart rate (bpm)",
                             60, 220, 150, key=f"hr_{tab_key}")
        oldpeak  = st.slider("ST depression (oldpeak)",
                             0.0, 7.0, 1.0, step=0.1, key=f"op_{tab_key}")
        fbs      = st.selectbox("Fasting blood sugar > 120 mg/dl",
                             ["No", "Yes"], key=f"fbs_{tab_key}")

    with c3:
        st.markdown('<div class="s-divider">Clinical Profile</div>',
                    unsafe_allow_html=True)
        cp       = st.selectbox("Chest pain type",
                             CHEST_PAIN_OPTIONS, key=f"cp_{tab_key}")
        restecg  = st.selectbox("Resting ECG",
                             ECG_OPTIONS, key=f"ecg_{tab_key}")
        exang    = st.selectbox("Exercise induced angina",
                             ["No", "Yes"], key=f"ex_{tab_key}")
        slope    = st.selectbox("ST segment slope",
                             SLOPE_OPTIONS, key=f"sl_{tab_key}")
        ca       = st.selectbox("Major vessels (0–3)",
                             [0, 1, 2, 3], key=f"ca_{tab_key}")
        thal     = st.selectbox("Thalassemia",
                             THAL_OPTIONS, key=f"th_{tab_key}")

    st.markdown('<div class="s-divider">Current Symptoms</div>',
                unsafe_allow_html=True)
    symptoms = st.multiselect(
        "Select all symptoms present",
        SYMPTOM_OPTIONS, key=f"sym_{tab_key}"
    )

    # ── Encode for ML model ────────────────────────────────────────
    sex_v   = 1 if sex == "Male" else 0
    cp_v    = CHEST_PAIN_OPTIONS.index(cp)
    fbs_v   = 1 if fbs == "Yes" else 0
    ecg_v   = ECG_OPTIONS.index(restecg)
    exang_v = 1 if exang == "Yes" else 0
    slope_v = SLOPE_OPTIONS.index(slope)
    thal_v  = THAL_OPTIONS.index(thal)
    abr     = age * trestbps / 1000

    encoded = pd.DataFrame([{
        'age'        : age,
        'sex'        : sex_v,
        'cp'         : cp_v,
        'trestbps'   : trestbps,
        'chol'       : chol,
        'fbs'        : fbs_v,
        'restecg'    : ecg_v,
        'thalach'    : thalach,
        'exang'      : exang_v,
        'oldpeak'    : oldpeak,
        'slope'      : slope_v,
        'ca'         : ca,
        'thal'       : thal_v,
        'age_bp_risk': abr,
    }])

    raw = {
        'age': age, 'sex': sex, 'bmi': bmi, 'smoking': smoking,
        'diabetes': diabetes, 'trestbps': trestbps, 'chol': chol,
        'thalach': thalach, 'oldpeak': oldpeak, 'fbs': fbs,
        'cp': cp, 'restecg': restecg, 'exang': exang,
        'slope': slope, 'ca': ca, 'thal': thal,
        'symptoms': symptoms,
    }

    return {'encoded': encoded, 'raw': raw}


def run_clinical_prediction(encoded_df: pd.DataFrame,
                             models: dict) -> dict:
    """
    Run ensemble prediction on encoded clinical features.
    Returns prediction dict from ensemble_predict().
    """
    from ensemble import ensemble_predict, get_model_disagreement_note
    result = ensemble_predict(encoded_df, models)
    result['disagreement_note'] = get_model_disagreement_note(result)
    return result


def symptom_risk_score(symptoms: list) -> dict:
    """
    Simple rule-based symptom risk scorer.
    Returns a score 0-100 and a label.
    """
    high_weight = {
        "Coughing blood"              : 30,
        "Shortness of breath"         : 20,
        "Chest pain / tightness"      : 20,
        "Recent surgery (< 3 months)" : 15,
        "Prolonged immobility (> 4 hours)": 15,
    }
    medium_weight = {
        "Leg swelling / pain"          : 10,
        "Rapid heart rate"             : 10,
        "Dizziness / fainting"         : 8,
        "Pregnancy or recent delivery" : 10,
        "Family history of clotting"   : 8,
    }
    score = 0
    for s in symptoms:
        score += high_weight.get(s, 0) + medium_weight.get(s, 0)
    score = min(score, 100)

    if score >= 50:
        label = "HIGH"
    elif score >= 25:
        label = "MEDIUM"
    elif score > 0:
        label = "LOW"
    else:
        label = "NONE"

    return {'score': score, 'label': label}