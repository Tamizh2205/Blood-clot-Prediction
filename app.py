import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
from report import generate_report

sys.path.append('src')
os.makedirs('reports', exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG + MEDICAL THEME CSS
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ClotAI — Diagnostic System",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&family=Inter:wght@400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    background-color: #0A1628 !important;
    color: #E8F0FE !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Top bar ── */
.top-bar {
    background: #0F1E36;
    border-bottom: 1px solid #1E3A5F;
    padding: 12px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 16px;
    font-weight: 600;
    color: #00D4FF;
    letter-spacing: 0.06em;
}
.status-badge {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid rgba(0, 200, 150, 0.4);
    color: #00C896;
    letter-spacing: 0.05em;
}

/* ── ECG strip ── */
.ecg-strip {
    background: #0F1E36;
    border-bottom: 1px solid #1E3A5F;
    padding: 8px 28px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 10px;
    color: #3A5A7A;
    letter-spacing: 0.1em;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0F1E36 !important;
    border-bottom: 1px solid #1E3A5F !important;
    padding: 0 20px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    color: #7A9CC8 !important;
    background: transparent !important;
    border: none !important;
    padding: 12px 18px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #00D4FF !important;
    border-bottom: 2px solid #00D4FF !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #0A1628 !important;
    padding: 24px 28px !important;
}

/* ── Cards ── */
.med-card {
    background: #122040;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 16px;
}
.med-card-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 10px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7A9CC8;
    margin-bottom: 12px;
}

/* ── Result banners ── */
.result-danger {
    background: rgba(230, 59, 74, 0.1);
    border: 1px solid rgba(230, 59, 74, 0.4);
    border-left: 4px solid #E63B4A;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
}
.result-safe {
    background: rgba(0, 200, 150, 0.08);
    border: 1px solid rgba(0, 200, 150, 0.3);
    border-left: 4px solid #00C896;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 12px 0;
}
.result-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 18px;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.result-danger .result-title { color: #E63B4A; }
.result-safe .result-title { color: #00C896; }
.result-sub { font-size: 12px; color: #7A9CC8; margin-top: 4px; }

/* ── Metric cards ── */
.metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 14px 0; }
.metric-box {
    background: #0F1E36;
    border: 1px solid #1E3A5F;
    border-radius: 8px;
    padding: 12px 14px;
}
.metric-box-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 10px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7A9CC8;
}
.metric-box-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 600;
    color: #E8F0FE;
    margin-top: 2px;
}

/* ── Section header ── */
.section-head {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #00D4FF;
    padding-bottom: 8px;
    border-bottom: 1px solid #1E3A5F;
    margin-bottom: 12px;
}

/* ── Sliders + inputs ── */
.stSlider > div > div > div { background: #1E3A5F !important; }
.stSlider > div > div > div > div { background: #00D4FF !important; }
.stSelectbox > div > div { background: #122040 !important; border-color: #1E3A5F !important; color: #E8F0FE !important; }
.stSlider label, .stSelectbox label { color: #7A9CC8 !important; font-size: 12px !important; }

/* ── Buttons ── */
.stButton > button {
    background: #00D4FF !important;
    color: #000 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    text-transform: uppercase !important;
}
.stButton > button:hover { background: #00bfe8 !important; }
.stDownloadButton > button {
    background: transparent !important;
    color: #00D4FF !important;
    border: 1px solid #00D4FF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 500 !important;
    font-size: 12px !important;
    letter-spacing: 0.06em !important;
    border-radius: 8px !important;
    text-transform: uppercase !important;
}

/* ── File uploader ── */
.stFileUploader > div {
    background: rgba(0, 212, 255, 0.03) !important;
    border: 1.5px dashed #1E3A5F !important;
    border-radius: 8px !important;
}
.stFileUploader label { color: #7A9CC8 !important; }

/* ── SHAP matplotlib style ── */
.stImage { border-radius: 8px; overflow: hidden; border: 1px solid #1E3A5F; }

/* ── Divider ── */
hr { border-color: #1E3A5F !important; }

/* ── Expanders ── */
.stExpander { background: #122040 !important; border: 1px solid #1E3A5F !important; border-radius: 8px !important; }
.stExpander summary { color: #00D4FF !important; font-family: 'Space Grotesk', sans-serif !important; }

/* ── Info / Warning / Error boxes ── */
.stAlert { border-radius: 8px !important; }

/* ── Progress bar ── */
.stProgress > div > div > div { background: #00D4FF !important; }

/* ── Warning override ── */
div[data-testid="stMarkdownContainer"] { color: #E8F0FE !important; }
</style>

<div class="top-bar">
    <div class="logo">⬡ &nbsp;CLOTAI — AI DIAGNOSTIC SYSTEM</div>
    <div class="status-badge">● SYSTEM ONLINE</div>
</div>
<div class="ecg-strip">LIVE MONITOR &nbsp;·&nbsp; 4 DIAGNOSTIC MODULES ACTIVE &nbsp;·&nbsp; MODELS LOADED</div>
""", unsafe_allow_html=True)

plt.rcParams.update({
    'figure.facecolor' : '#0A1628',
    'axes.facecolor'   : '#0F1E36',
    'axes.edgecolor'   : '#1E3A5F',
    'axes.labelcolor'  : '#7A9CC8',
    'xtick.color'      : '#7A9CC8',
    'ytick.color'      : '#7A9CC8',
    'text.color'       : '#E8F0FE',
    'grid.color'       : '#1E3A5F',
    'grid.alpha'       : 0.5,
})

# ══════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_risk_model():
    return joblib.load('models/xgboost.pkl')

@st.cache_resource
def load_ct():
    from ct_detector import load_ct_model
    return load_ct_model()

@st.cache_resource
def load_mri():
    from mri_detector import load_mri_model
    return load_mri_model()

@st.cache_resource
def load_us():
    from ultrasound_detector import load_ultrasound_model
    return load_ultrasound_model()

risk_model = load_risk_model()

# ══════════════════════════════════════════════════════════════════
# HELPER: display result banner
# ══════════════════════════════════════════════════════════════════
def show_result_banner(label, confidence, risk_level, note):
    is_danger = risk_level in ('HIGH', 'MEDIUM')
    cls = 'result-danger' if is_danger else 'result-safe'
    icon = '⚠' if risk_level == 'HIGH' else ('△' if risk_level == 'MEDIUM' else '✓')
    st.markdown(f"""
    <div class="{cls}">
        <div class="result-title">{icon} &nbsp;{label}</div>
        <div class="result-sub">Confidence: {confidence}% &nbsp;·&nbsp; Risk level: {risk_level}</div>
        <div class="result-sub" style="margin-top:6px">{note}</div>
    </div>""", unsafe_allow_html=True)

def show_metrics(conf, risk, model_name, time_str):
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-box"><div class="metric-box-label">Confidence</div>
            <div class="metric-box-val">{conf}%</div></div>
        <div class="metric-box"><div class="metric-box-label">Risk Level</div>
            <div class="metric-box-val" style="color:{'#E63B4A' if risk=='HIGH' else ('#F59E0B' if risk=='MEDIUM' else '#00C896')}">{risk}</div></div>
        <div class="metric-box"><div class="metric-box-label">Model</div>
            <div class="metric-box-val" style="font-size:13px">{model_name}</div></div>
        <div class="metric-box"><div class="metric-box-label">Inference</div>
            <div class="metric-box-val">{time_str}</div></div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🫁  CT SCAN — PULMONARY EMBOLISM",
    "🧠  MRI — BRAIN CLOT / STROKE",
    "🔊  ULTRASOUND — DVT DETECTION",
    "📋  CLINICAL RISK PREDICTOR"
])

# ── TAB 1: CT ─────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-head">CT Scan Analysis — Pulmonary Embolism Detection</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:12px;color:#7A9CC8;margin-bottom:20px">Upload a chest CT scan. EfficientNet-B0 will detect signs of pulmonary embolism with confidence scoring.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown('<div class="med-card-label">Upload CT Image</div>', unsafe_allow_html=True)
        ct_file = st.file_uploader("", type=['png','jpg','jpeg'], key="ct", label_visibility="collapsed")
        if ct_file:
            ct_img = Image.open(ct_file).convert('RGB')
            st.image(ct_img, caption="Uploaded CT Scan", use_column_width=True)

    with col2:
        st.markdown('<div class="med-card-label">Analysis Result</div>', unsafe_allow_html=True)
        if ct_file:
            if st.button("ANALYZE CT SCAN", key="ct_btn", use_container_width=True):
                with st.spinner("Running EfficientNet-B0 inference..."):
                    import time
                    t0 = time.time()
                    try:
                        ct_model = load_ct()
                        from ct_detector import predict_ct
                        result = predict_ct(ct_img, ct_model)
                        elapsed = f"{time.time()-t0:.2f}s"
                        show_result_banner(result['label'], result['confidence'],
                            result['risk_level'],
                            "Immediate clinical evaluation recommended." if result['risk_level']=='HIGH'
                            else "Monitor and follow up with physician.")
                        show_metrics(result['confidence'], result['risk_level'], "EfficientNet-B0", elapsed)
                        st.progress(result['raw_prob'])
                    except Exception as e:
                        st.error(f"Model not found. Train ct_model.pth first.\n{e}")
        else:
            st.markdown('<p style="font-size:12px;color:#3A5A7A;margin-top:40px;text-align:center">Upload a CT scan image to begin analysis</p>', unsafe_allow_html=True)

    with st.expander("What is Pulmonary Embolism?"):
        st.markdown('<p style="font-size:12px;color:#7A9CC8;line-height:1.7">A pulmonary embolism (PE) is a blood clot that travels to and blocks an artery in the lungs. It can be life-threatening if not treated quickly. Common symptoms include chest pain, shortness of breath, and rapid heart rate. CT pulmonary angiography (CTPA) is the gold standard for diagnosis.</p>', unsafe_allow_html=True)

# ── TAB 2: MRI ────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-head">Brain MRI Analysis — Clot / Stroke Indicator</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:12px;color:#7A9CC8;margin-bottom:20px">Upload a brain MRI scan. ResNet50 will detect anomalies associated with brain clots and stroke indicators.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown('<div class="med-card-label">Upload MRI Image</div>', unsafe_allow_html=True)
        mri_file = st.file_uploader("", type=['png','jpg','jpeg'], key="mri", label_visibility="collapsed")
        if mri_file:
            mri_img = Image.open(mri_file).convert('RGB')
            st.image(mri_img, caption="Uploaded MRI Scan", use_column_width=True)

    with col2:
        st.markdown('<div class="med-card-label">Analysis Result</div>', unsafe_allow_html=True)
        if mri_file:
            if st.button("ANALYZE MRI SCAN", key="mri_btn", use_container_width=True):
                with st.spinner("Running ResNet50 inference..."):
                    import time
                    t0 = time.time()
                    try:
                        mri_model = load_mri()
                        from mri_detector import predict_mri
                        result = predict_mri(mri_img, mri_model)
                        elapsed = f"{time.time()-t0:.2f}s"
                        show_result_banner(result['label'], result['confidence'],
                            result['risk_level'],
                            "Neurology consult recommended immediately." if result['risk_level']=='HIGH'
                            else "Follow-up MRI in 3 months advised.")
                        show_metrics(result['confidence'], result['risk_level'], "ResNet-50", elapsed)
                        st.progress(result['raw_prob'])
                    except Exception as e:
                        st.error(f"Model not found. Train mri_model.pth first.\n{e}")
        else:
            st.markdown('<p style="font-size:12px;color:#3A5A7A;margin-top:40px;text-align:center">Upload an MRI scan image to begin analysis</p>', unsafe_allow_html=True)

    with st.expander("What are brain clots?"):
        st.markdown('<p style="font-size:12px;color:#7A9CC8;line-height:1.7">Brain clots can cause ischemic strokes, where blood flow to part of the brain is blocked. Early detection through MRI imaging allows faster treatment with clot-dissolving medications (tPA) or mechanical thrombectomy. Time to treatment is critical — every minute counts.</p>', unsafe_allow_html=True)

# ── TAB 3: ULTRASOUND ─────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-head">Ultrasound Analysis — Deep Vein Thrombosis (DVT)</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:12px;color:#7A9CC8;margin-bottom:20px">Upload an ultrasound image. EfficientNet-B2 will detect DVT clot patterns in venous tissue.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown('<div class="med-card-label">Upload Ultrasound Image</div>', unsafe_allow_html=True)
        us_file = st.file_uploader("", type=['png','jpg','jpeg'], key="us", label_visibility="collapsed")
        if us_file:
            us_img = Image.open(us_file).convert('RGB')
            st.image(us_img, caption="Uploaded Ultrasound", use_column_width=True)

    with col2:
        st.markdown('<div class="med-card-label">Analysis Result</div>', unsafe_allow_html=True)
        if us_file:
            if st.button("ANALYZE ULTRASOUND", key="us_btn", use_container_width=True):
                with st.spinner("Running EfficientNet-B2 inference..."):
                    import time
                    t0 = time.time()
                    try:
                        us_model = load_us()
                        from ultrasound_detector import predict_ultrasound
                        result = predict_ultrasound(us_img, us_model)
                        elapsed = f"{time.time()-t0:.2f}s"
                        show_result_banner(result['label'], result['confidence'],
                            result['risk_level'],
                            "Anticoagulation therapy consultation needed." if result['risk_level']=='HIGH'
                            else "Compression stockings and monitoring advised.")
                        show_metrics(result['confidence'], result['risk_level'], "EfficientNet-B2", elapsed)
                        st.progress(result['raw_prob'])
                    except Exception as e:
                        st.error(f"Model not found. Train ultrasound_model.pth first.\n{e}")
        else:
            st.markdown('<p style="font-size:12px;color:#3A5A7A;margin-top:40px;text-align:center">Upload an ultrasound image to begin analysis</p>', unsafe_allow_html=True)

    with st.expander("What is DVT?"):
        st.markdown('<p style="font-size:12px;color:#7A9CC8;line-height:1.7">Deep Vein Thrombosis (DVT) is a blood clot that forms in a deep vein, usually in the legs. If untreated, the clot can break loose and travel to the lungs, causing a pulmonary embolism. Risk factors include prolonged immobility, surgery, pregnancy, and certain genetic conditions.</p>', unsafe_allow_html=True)

# ── TAB 4: CLINICAL RISK ──────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-head">Clinical Risk Assessment — XGBoost + SHAP Explainability</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:12px;color:#7A9CC8;margin-bottom:20px">Enter patient clinical data for AI-powered clot risk prediction with full SHAP explanation and downloadable PDF report.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="med-card-label">Vital Signs & Labs</div>', unsafe_allow_html=True)
        age      = st.slider("Age (years)", 20, 90, 50)
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol     = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
        thalach  = st.slider("Maximum Heart Rate (bpm)", 60, 220, 150)
        oldpeak  = st.slider("ST Depression (Oldpeak)", 0.0, 7.0, 1.0, step=0.1)

    with col2:
        st.markdown('<div class="med-card-label">Patient Profile</div>', unsafe_allow_html=True)
        sex     = st.selectbox("Sex", ["Male", "Female"])
        cp      = st.selectbox("Chest Pain Type", ["Typical Angina","Atypical Angina","Non-Anginal","Asymptomatic"])
        fbs     = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No","Yes"])
        restecg = st.selectbox("Resting ECG", ["Normal","ST-T Abnormality","Left Ventricular Hypertrophy"])
        exang   = st.selectbox("Exercise Induced Angina", ["No","Yes"])
        slope   = st.selectbox("ST Segment Slope", ["Upsloping","Flat","Downsloping"])
        ca      = st.selectbox("Major Vessels (0–3)", [0,1,2,3])
        thal    = st.selectbox("Thalassemia", ["Normal","Fixed Defect","Reversible Defect"])

    sex_val     = 1 if sex=="Male" else 0
    cp_val      = ["Typical Angina","Atypical Angina","Non-Anginal","Asymptomatic"].index(cp)
    fbs_val     = 1 if fbs=="Yes" else 0
    restecg_val = ["Normal","ST-T Abnormality","Left Ventricular Hypertrophy"].index(restecg)
    exang_val   = 1 if exang=="Yes" else 0
    slope_val   = ["Upsloping","Flat","Downsloping"].index(slope)
    thal_val    = ["Normal","Fixed Defect","Reversible Defect"].index(thal)
    age_bp_risk = age * trestbps / 1000

    input_data = pd.DataFrame([{
        'age':age,'sex':sex_val,'cp':cp_val,'trestbps':trestbps,
        'chol':chol,'fbs':fbs_val,'restecg':restecg_val,'thalach':thalach,
        'exang':exang_val,'oldpeak':oldpeak,'slope':slope_val,'ca':ca,
        'thal':thal_val,'age_bp_risk':age_bp_risk
    }])

    st.markdown("---")
    if st.button("RUN RISK ASSESSMENT", use_container_width=True):
        prediction  = risk_model.predict(input_data)[0]
        probability = risk_model.predict_proba(input_data)[0][1]
        confidence  = round(probability * 100, 1)

        show_result_banner(
            "HIGH CLOT RISK DETECTED" if prediction==1 else "LOW CLOT RISK",
            confidence,
            "HIGH" if prediction==1 else "LOW",
            "Consult a physician immediately for further evaluation." if prediction==1
            else "Maintain healthy lifestyle. Routine checkups advised."
        )
        show_metrics(confidence, "HIGH" if prediction==1 else "LOW", "XGBoost", "< 0.1s")
        st.progress(float(probability))

        st.markdown("---")
        st.markdown('<div class="section-head">AI Explanation — SHAP Feature Analysis</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:12px;color:#7A9CC8;margin-bottom:12px">Red bars increase clot risk · Blue bars decrease risk · Bar length = impact magnitude</p>', unsafe_allow_html=True)

        explainer   = shap.Explainer(risk_model)
        shap_values = explainer(input_data)
        shap_path   = 'reports/shap_temp.png'

        fig, _ = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_values[0], show=False)
        plt.tight_layout()
        plt.savefig(shap_path, bbox_inches='tight', dpi=150,
                    facecolor='#0A1628', edgecolor='none')
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        st.markdown('<div class="section-head">Patient Summary</div>', unsafe_allow_html=True)
        summary = pd.DataFrame({
            "Parameter" : ["Age","Sex","Blood Pressure","Cholesterol","Max Heart Rate","Chest Pain","ST Depression"],
            "Value"     : [f"{age} yrs", sex, f"{trestbps} mm Hg", f"{chol} mg/dl",
                           f"{thalach} bpm", cp, str(oldpeak)],
            "Status"    : [
                "Elevated" if age>55 else "Normal",
                "—",
                "High" if trestbps>140 else "Normal",
                "High" if chol>240 else "Normal",
                "Low" if thalach<100 else "Normal",
                cp, "Elevated" if oldpeak>2 else "Normal"
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.markdown("---")
        patient_data = {
            'age':age,'sex':sex,'trestbps':trestbps,'chol':chol,
            'thalach':thalach,'fbs':fbs,'cp':cp,'exang':exang,
            'oldpeak':oldpeak,'slope':slope,'ca':ca,'thal':thal
        }
        report_path = generate_report(
            patient_data=patient_data, prediction=int(prediction),
            confidence=confidence, shap_plot_path=shap_path
        )
        with open(report_path,'rb') as f:
            st.download_button(
                label="DOWNLOAD PDF REPORT",
                data=f,
                file_name="ClotAI_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )

st.markdown('<p style="text-align:center;font-size:11px;color:#3A5A7A;padding:20px;font-family:Space Grotesk,sans-serif;letter-spacing:0.06em">CLOTAI DIAGNOSTIC SYSTEM &nbsp;·&nbsp; FOR EDUCATIONAL PURPOSES ONLY &nbsp;·&nbsp; NOT A SUBSTITUTE FOR MEDICAL ADVICE</p>', unsafe_allow_html=True)