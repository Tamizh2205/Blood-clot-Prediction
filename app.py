import streamlit as st
import pandas as pd
import numpy as np
import joblib, shap, os, sys, time, datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('src')
os.makedirs('reports', exist_ok=True)
os.makedirs('plots',   exist_ok=True)
os.makedirs('data',    exist_ok=True)

from report          import generate_combined_report, generate_report
from history         import log_prediction, get_all_history
from audit           import log_audit, get_audit_log, export_audit_csv
from ensemble        import load_ensemble, ensemble_predict, get_model_disagreement_note
from analytics       import (compute_metrics, plot_roc_comparison,
                              plot_precision_recall, plot_confusion_matrix,
                              plot_feature_importance)
from clinical_panel  import (render_clinical_form, run_clinical_prediction,
                              symptom_risk_score)
from combined_analysis import fuse_results, get_risk_color, get_risk_kind


from datetime import datetime
import random

def generate_patient_id():
    return f"PT-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000,9999)}"

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="HEMO CHECK — Blood Clot Diagnostic System",
    page_icon="assets/favicon.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

plt.rcParams.update({
    'figure.facecolor':'#ffffff','axes.facecolor':'#f8fafc',
    'axes.edgecolor':'#e2e8f0','axes.labelcolor':'#475569',
    'xtick.color':'#64748b','ytick.color':'#64748b',
    'text.color':'#1e293b','grid.color':'#e2e8f0','grid.alpha':0.8,
    'axes.spines.top':False,'axes.spines.right':False,
})

# ══════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Serif+Display&display=swap');
@import url('https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[class*="css"]{
  background:#f1f5f9 !important;
  color:#1e293b !important;
  font-family:'DM Sans',sans-serif !important;
}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:0 !important;max-width:100% !important}
section[data-testid="stSidebar"]{background:#fff !important;border-right:1px solid #e2e8f0 !important}

/* topbar */
.topbar{background:#fff;border-bottom:1px solid #e2e8f0;padding:0 40px;height:64px;
  display:flex;align-items:center;justify-content:space-between}
.brand{display:flex;align-items:center;gap:12px}
.brand-name{font-family:'DM Serif Display',serif;font-size:20px;color:#0f172a;letter-spacing:-.02em}
.brand-tag{font-size:10px;color:#0891b2;letter-spacing:.08em;text-transform:uppercase;font-weight:500;margin-top:1px}
.nav-links{display:flex;gap:4px}
.nav-link{font-size:13px;font-weight:500;color:#475569;padding:6px 14px;
  border-radius:8px;cursor:pointer}
.nav-link:hover{background:#f1f5f9;color:#0f172a}
.nav-link.active{background:#ecfeff;color:#0891b2}
.nav-badge{display:inline-block;font-size:9px;font-weight:600;padding:2px 7px;
  border-radius:20px;background:#ecfeff;color:#0e7490;border:1px solid #a5f3fc;margin-left:4px}

/* hero */
.hero{background:linear-gradient(135deg,#0c4a6e 0%,#0369a1 40%,#0891b2 100%);
  padding:64px 40px 56px;position:relative;overflow:hidden}
.hero::before{content:'';position:absolute;inset:0;
  background:radial-gradient(circle at 70% 50%,rgba(255,255,255,.06) 0%,transparent 60%)}
.hero-eyebrow{display:inline-flex;align-items:center;gap:6px;
  background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.2);
  color:#bae6fd;font-size:11px;font-weight:500;letter-spacing:.06em;
  padding:5px 14px;border-radius:20px;margin-bottom:20px;text-transform:uppercase}
.hero-title{font-family:'DM Serif Display',serif;font-size:44px;color:#fff;
  line-height:1.2;letter-spacing:-.02em;margin-bottom:16px}
.hero-title span{color:#7dd3fc}
.hero-desc{font-size:15px;color:rgba(255,255,255,.78);line-height:1.7;
  max-width:520px;margin-bottom:32px}
.hero-btns{display:flex;gap:12px;flex-wrap:wrap}
.hero-btn-p{display:inline-flex;align-items:center;gap:8px;background:#fff;
  color:#0369a1;font-size:13px;font-weight:600;padding:11px 24px;
  border-radius:10px;border:none;cursor:pointer;font-family:'DM Sans',sans-serif}
.hero-btn-s{display:inline-flex;align-items:center;gap:8px;
  background:rgba(255,255,255,.12);color:#fff;font-size:13px;font-weight:500;
  padding:11px 24px;border-radius:10px;border:1px solid rgba(255,255,255,.25);
  cursor:pointer;font-family:'DM Sans',sans-serif}
.hero-stats{display:flex;gap:40px;margin-top:48px;padding-top:32px;
  border-top:1px solid rgba(255,255,255,.15)}
.hs-num{font-family:'DM Serif Display',serif;font-size:28px;color:#fff;line-height:1}
.hs-lbl{font-size:11px;color:rgba(255,255,255,.6);letter-spacing:.04em;margin-top:4px}

/* stat strip */
.stat-strip{background:#0369a1;padding:40px;display:grid;
  grid-template-columns:repeat(4,1fr);gap:0}
.stat-item{text-align:center;padding:0 20px;border-right:1px solid rgba(255,255,255,.15)}
.stat-item:last-child{border-right:none}
.stat-num{font-family:'DM Serif Display',serif;font-size:36px;color:#fff}
.stat-lbl{font-size:12px;color:rgba(255,255,255,.65);letter-spacing:.04em;margin-top:4px}

/* sections */
.section{padding:56px 40px}
.section-alt{background:#fff;padding:56px 40px}
.section-label{font-size:11px;font-weight:600;letter-spacing:.1em;
  text-transform:uppercase;color:#0891b2;margin-bottom:8px}
.section-title{font-family:'DM Serif Display',serif;font-size:30px;color:#0f172a;
  letter-spacing:-.02em;margin-bottom:10px;line-height:1.3}
.section-desc{font-size:14px;color:#64748b;max-width:560px;line-height:1.7;margin-bottom:40px}

/* feature cards */
.feat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:20px}
.feat-card{background:#fff;border:1px solid #e2e8f0;border-radius:16px;
  padding:28px 24px;transition:border-color .2s,transform .2s}
.feat-card:hover{border-color:#7dd3fc;transform:translateY(-2px)}
.feat-icon-wrap{width:48px;height:48px;border-radius:12px;display:flex;
  align-items:center;justify-content:center;font-size:22px;margin-bottom:16px}
.fi-teal{background:#ecfeff;color:#0891b2}
.fi-blue{background:#eff6ff;color:#2563eb}
.fi-green{background:#f0fdf4;color:#16a34a}
.fi-amber{background:#fffbeb;color:#d97706}
.fi-red{background:#fff1f2;color:#e11d48}
.fi-purple{background:#faf5ff;color:#9333ea}
.feat-title{font-size:15px;font-weight:600;color:#0f172a;margin-bottom:8px}
.feat-desc{font-size:13px;color:#64748b;line-height:1.6}

/* module cards */
.module-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:20px}
.module-card{background:#fff;border:1px solid #e2e8f0;border-radius:16px;
  padding:28px;display:flex;gap:20px;align-items:flex-start}
.module-icon{width:56px;height:56px;border-radius:14px;display:flex;
  align-items:center;justify-content:center;font-size:26px;flex-shrink:0}
.module-title{font-size:16px;font-weight:600;color:#0f172a;margin-bottom:4px}
.module-model{font-size:11px;color:#0891b2;font-weight:500;letter-spacing:.04em;margin-bottom:8px}
.module-desc{font-size:13px;color:#64748b;line-height:1.6}

/* info banner */
.info-bar{background:#eff6ff;border:1px solid #bfdbfe;border-left:4px solid #2563eb;
  border-radius:10px;padding:12px 16px;margin-bottom:20px;
  font-size:12px;color:#1e40af;line-height:1.6}

/* tabs */
.stTabs [data-baseweb="tab-list"]{background:#fff !important;
  border-bottom:1px solid #e2e8f0 !important;padding:0 40px !important;gap:0 !important}
.stTabs [data-baseweb="tab"]{font-family:'DM Sans',sans-serif !important;
  font-size:13px !important;font-weight:500 !important;color:#64748b !important;
  background:transparent !important;border:none !important;
  padding:14px 18px !important;border-bottom:2px solid transparent !important}
.stTabs [aria-selected="true"]{color:#0891b2 !important;border-bottom-color:#0891b2 !important}
.stTabs [data-baseweb="tab-panel"]{background:#f1f5f9 !important;padding:32px 40px !important}

/* result banners */
.r-danger{background:#fff1f2;border:1px solid #fecdd3;border-left:4px solid #e11d48;
  border-radius:10px;padding:14px 18px;margin:12px 0}
.r-safe{background:#f0fdf4;border:1px solid #bbf7d0;border-left:4px solid #16a34a;
  border-radius:10px;padding:14px 18px;margin:12px 0}
.r-warn{background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #d97706;
  border-radius:10px;padding:14px 18px;margin:12px 0}
.r-neutral{background:#f8fafc;border:1px solid #e2e8f0;border-left:4px solid #94a3b8;
  border-radius:10px;padding:14px 18px;margin:12px 0}
.r-combined{background:linear-gradient(135deg,#eff6ff 0%,#ecfeff 100%);
  border:2px solid #7dd3fc;border-radius:12px;padding:20px 22px;margin:16px 0}
.r-title{font-size:16px;font-weight:600}
.r-danger  .r-title{color:#be123c}
.r-safe    .r-title{color:#15803d}
.r-warn    .r-title{color:#92400e}
.r-neutral .r-title{color:#475569}
.r-combined .r-title{color:#0369a1;font-size:18px}
.r-sub{font-size:12px;color:#64748b;margin-top:4px;line-height:1.5}

/* metric grid */
.m-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:14px 0}
.m-grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin:14px 0}
.m-box{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:14px 16px}
.m-lbl{font-size:10px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#94a3b8}
.m-val{font-size:22px;font-weight:600;margin-top:4px;color:#0f172a}

/* contribution bar */
.contrib-wrap{background:#fff;border:1px solid #e2e8f0;border-radius:10px;
  padding:16px 18px;margin:14px 0}
.contrib-title{font-size:11px;font-weight:600;letter-spacing:.08em;
  text-transform:uppercase;color:#94a3b8;margin-bottom:12px}
.contrib-row{margin-bottom:10px}
.contrib-label{display:flex;justify-content:space-between;font-size:12px;
  color:#475569;margin-bottom:4px}
.contrib-bar-bg{height:8px;background:#f1f5f9;border-radius:4px;overflow:hidden}
.contrib-bar-fill{height:100%;border-radius:4px;transition:width .8s ease}

/* scan summary */
.scan-summary{background:#fff;border:1px solid #e2e8f0;border-radius:12px;
  padding:20px 22px;margin-top:16px}
.ss-head{font-size:13px;font-weight:600;color:#0f172a;margin-bottom:12px;
  display:flex;align-items:center;gap:8px}
.ss-row{display:flex;justify-content:space-between;align-items:center;
  padding:7px 0;border-bottom:1px solid #f1f5f9;font-size:12px}
.ss-row:last-child{border-bottom:none}
.ss-key{color:#64748b}
.ss-val{font-weight:500;color:#1e293b}
.ss-val.danger{color:#be123c}
.ss-val.safe{color:#15803d}
.ss-val.warn{color:#92400e}

/* rec list */
.rec-list{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:18px 20px;margin-top:14px}
.rec-head{font-size:13px;font-weight:600;color:#0f172a;margin-bottom:12px}
.rec-item{display:flex;gap:10px;align-items:flex-start;padding:6px 0;
  border-bottom:1px solid #f8fafc;font-size:12px;color:#475569;line-height:1.5}
.rec-item:last-child{border-bottom:none}
.rec-num{width:20px;height:20px;border-radius:50%;background:#ecfeff;
  color:#0891b2;font-size:10px;font-weight:700;display:flex;align-items:center;
  justify-content:center;flex-shrink:0;margin-top:1px}

/* section divider */
.s-divider{font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  color:#0891b2;padding-bottom:8px;border-bottom:1px solid #e2e8f0;margin:20px 0 14px}

/* disclaimer box */
.disclaimer{background:#eff6ff;border:1px solid #bfdbfe;border-radius:12px;
  padding:16px 20px;display:flex;gap:12px;align-items:flex-start}
.disc-icon{font-size:18px;color:#2563eb;margin-top:1px;flex-shrink:0}
.disc-title{font-size:13px;font-weight:600;color:#1e40af;margin-bottom:4px}
.disc-body{font-size:12px;color:#3b82f6;line-height:1.6}

/* patient info */
.pt-block{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
  padding:14px 16px;margin-bottom:12px}
.pt-label{font-size:10px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
  color:#94a3b8;margin-bottom:10px}
.pt-row{display:flex;justify-content:space-between;padding:5px 0;
  border-bottom:1px solid #f1f5f9;font-size:12px}
.pt-row:last-child{border-bottom:none}
.pt-k{color:#64748b}
.pt-v{font-weight:500;color:#1e293b}

/* footer */
.site-footer{background:#0f172a;padding:32px 40px;display:flex;
  align-items:center;justify-content:space-between}
.sf-brand{font-family:'DM Serif Display',serif;font-size:18px;color:#fff}
.sf-tag{font-size:10px;color:#475569;letter-spacing:.06em;margin-top:2px}
.sf-copy{font-size:11px;color:#475569}

/* inputs */
.stSlider>div>div>div{background:#e2e8f0 !important}
.stSlider>div>div>div>div{background:#0891b2 !important}
.stSelectbox>div>div{background:#fff !important;border-color:#e2e8f0 !important;
  color:#1e293b !important;border-radius:10px !important}
.stTextInput>div>div>input{background:#fff !important;border-color:#e2e8f0 !important;
  color:#1e293b !important;border-radius:10px !important}
.stTextArea>div>div>textarea{background:#fff !important;border-color:#e2e8f0 !important;
  color:#1e293b !important;border-radius:10px !important}
.stSlider label,.stSelectbox label,.stTextInput label,.stTextArea label,.stMultiSelect label{
  color:#475569 !important;font-size:12px !important;font-weight:500 !important}
.stMultiSelect>div>div{background:#fff !important;border-color:#e2e8f0 !important;border-radius:10px !important}

/* buttons */
.stButton>button{background:#0891b2 !important;color:#fff !important;
  font-family:'DM Sans',sans-serif !important;font-weight:600 !important;
  font-size:13px !important;border:none !important;border-radius:10px !important;
  padding:10px 20px !important}
.stButton>button:hover{background:#0e7490 !important}
.stDownloadButton>button{background:#fff !important;color:#0891b2 !important;
  border:1px solid #a5f3fc !important;font-family:'DM Sans',sans-serif !important;
  font-weight:500 !important;font-size:13px !important;border-radius:10px !important}

/* file uploader */
.stFileUploader>div{background:#f8fafc !important;
  border:2px dashed #cbd5e1 !important;border-radius:12px !important}

/* logo */
.brand-logo{
    width:45px;
    height:45px;
    object-fit:contain;
    border-radius:8px;
}

/* dataframe */
[data-testid="stDataFrameContainer"]{background:#fff !important;
  border:1px solid #e2e8f0 !important;border-radius:12px !important;overflow:hidden !important}

/* expander */
.stExpander{background:#fff !important;border:1px solid #e2e8f0 !important;border-radius:12px !important}
.stExpander summary{color:#0891b2 !important;font-family:'DM Sans',sans-serif !important;
  font-size:13px !important;font-weight:500 !important}

/* image */
[data-testid="stImage"] img{border-radius:12px;border:1px solid #e2e8f0}

/* progress */
.stProgress>div>div>div{background:#0891b2 !important}

div[data-testid="stMarkdownContainer"]{color:#1e293b !important}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# SVG LOGO
# ══════════════════════════════════════════════════════════════════
LOGO_SVG = """<svg width="40" height="40" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="40" height="40" rx="10" fill="#0369a1"/>
  <path d="M20 8C20 8 14 12 14 18C14 22 17 25 20 26C23 25 26 22 26 18C26 12 20 8 20 8Z" fill="#7dd3fc" opacity=".9"/>
  <path d="M12 22L16 18L19 22L21 16L24 22L28 18" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="20" cy="31" r="2.5" fill="#38bdf8"/>
</svg>"""

# ══════════════════════════════════════════════════════════════════
# TOPBAR
# ══════════════════════════════════════════════════════════════════
import base64

def get_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo = get_base64("assets/favicon.png")

st.markdown(f"""
<div class="topbar">
  <div class="brand">
    <img src="data:image/png;base64,{logo}" class="brand-logo">
    <div>
      <div class="brand-name">HEMO CHECK</div>
      <div class="brand-tag">Blood Clot Diagnostic Intelligence System</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_ensemble_models():
    return load_ensemble()

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

models = load_ensemble_models()

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
def sh(title):
    st.markdown(f'<div class="s-divider">{title}</div>', unsafe_allow_html=True)

def banner(label, sub, kind='danger'):
    cls  = {'danger':'r-danger','safe':'r-safe','warn':'r-warn','neutral':'r-neutral'}[kind]
    st.markdown(f'<div class="{cls}"><div class="r-title">{label}</div>'
                f'<div class="r-sub">{sub}</div></div>', unsafe_allow_html=True)

def combined_banner(label, sub, conf, color):
    st.markdown(f"""
    <div class="r-combined">
      <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <div>
          <div style="font-size:11px;font-weight:600;letter-spacing:.08em;
                      text-transform:uppercase;color:#0891b2;margin-bottom:6px">
            Combined AI Assessment
          </div>
          <div class="r-title" style="color:{color};font-size:20px">{label}</div>
          <div class="r-sub" style="margin-top:6px">{sub}</div>
        </div>
        <div style="text-align:center;background:#fff;border-radius:12px;
                    padding:12px 20px;border:2px solid {color};min-width:90px">
          <div style="font-size:28px;font-weight:700;color:{color};line-height:1">{conf}%</div>
          <div style="font-size:10px;color:#64748b;margin-top:3px;letter-spacing:.06em">CONFIDENCE</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

def metric_grid(items, cols=2):
    cls = "m-grid" if cols == 2 else "m-grid-3"
    cells = ''.join([
        f'<div class="m-box"><div class="m-lbl">{l}</div>'
        f'<div class="m-val" style="color:{c}">{v}</div></div>'
        for l,v,c in items
    ])
    st.markdown(f'<div class="{cls}">{cells}</div>', unsafe_allow_html=True)

def contribution_bars(scan_c, clinical_c, symptom_c):
    st.markdown(f"""
    <div class="contrib-wrap">
      <div class="contrib-title">Score Contribution Breakdown</div>
      <div class="contrib-row">
        <div class="contrib-label">
          <span>Scan model (50%)</span><span>{scan_c}%</span>
        </div>
        <div class="contrib-bar-bg">
          <div class="contrib-bar-fill" style="width:{min(scan_c*2,100)}%;background:#0891b2"></div>
        </div>
      </div>
      <div class="contrib-row">
        <div class="contrib-label">
          <span>Clinical model (35%)</span><span>{clinical_c}%</span>
        </div>
        <div class="contrib-bar-bg">
          <div class="contrib-bar-fill" style="width:{min(clinical_c*2,100)}%;background:#0369a1"></div>
        </div>
      </div>
      <div class="contrib-row" style="margin-bottom:0">
        <div class="contrib-label">
          <span>Symptom score (15%)</span><span>{symptom_c}%</span>
        </div>
        <div class="contrib-bar-bg">
          <div class="contrib-bar-fill" style="width:{min(symptom_c*2,100)}%;background:#7dd3fc"></div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

def scan_summary(title, rows):
    rows_html = ''.join([
        f'<div class="ss-row"><span class="ss-key">{k}</span>'
        f'<span class="ss-val {c}">{v}</span></div>'
        for k,v,c in rows
    ])
    st.markdown(f"""
    <div class="scan-summary">
      <div class="ss-head">{title}</div>
      {rows_html}
    </div>""", unsafe_allow_html=True)

def rec_list(recs):
    items = ''.join([
        f'<div class="rec-item"><div class="rec-num">{i}</div><div>{r}</div></div>'
        for i,r in enumerate(recs, 1)
    ])
    st.markdown(f"""
    <div class="rec-list">
      <div class="rec-head">Clinical Recommendations</div>
      {items}
    </div>""", unsafe_allow_html=True)

def pt_info_block(pid, scan_type, phy, date):
    st.markdown(f"""
    <div class="pt-block">
      <div class="pt-label">Patient Session</div>
      <div class="pt-row"><span class="pt-k">Patient ID</span><span class="pt-v">{pid}</span></div>
      <div class="pt-row"><span class="pt-k">Scan type</span><span class="pt-v">{scan_type}</span></div>
      <div class="pt-row"><span class="pt-k">Physician</span><span class="pt-v">{phy}</span></div>
      <div class="pt-row"><span class="pt-k">Date</span><span class="pt-v">{date}</span></div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px;font-size:11px;font-weight:600;
                letter-spacing:.08em;text-transform:uppercase;color:#0891b2">
      Patient Session
    </div>""", unsafe_allow_html=True)
    default_pid = f"PT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    pid    = st.text_input("Patient ID",  value=default_pid)
    phy    = st.text_input("Physician",   value="Dr. Sharma")
    notes  = st.text_area("Clinical Notes",
                          placeholder="Optional notes for the report...",
                          height=80)
    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px;color:#94a3b8;line-height:1.6">
      For educational purposes only.<br>Not for clinical diagnosis.
    </div>""", unsafe_allow_html=True)

today = datetime.today().strftime("%d %b %Y")


# ══════════════════════════════════════════════════════════════════
# SHARED SCAN+CLINICAL RUNNER
# ══════════════════════════════════════════════════════════════════
def run_combined_scan_tab(img_file, model_loader, predictor_fn,
                           model_type, scan_label, model_name,
                           clinical_data, pid, phy, notes):
    """
    Full combined pipeline:
    1. Run scan model
    2. Run clinical ensemble
    3. Run symptom scorer
    4. Fuse all three
    5. Display unified result
    6. Generate PDF
    """
    img_pil = Image.open(img_file).convert('RGB')

    with st.spinner(f"Running {model_name} + clinical ensemble..."):
        try:
            # ── Scan inference ──────────────────────────────────────
            t0      = time.time()
            model   = model_loader()
            scan_res= predictor_fn(img_pil, model)
            elapsed = round(time.time() - t0, 2)
            scan_res['elapsed']    = f"{elapsed}s"
            scan_res['model_name'] = model_name

            # ── Clinical inference ──────────────────────────────────
            clin_res = run_clinical_prediction(
                clinical_data['encoded'], models)

            # ── Symptom score ───────────────────────────────────────
            symp_res = symptom_risk_score(
                clinical_data['raw'].get('symptoms', []))

            # ── Fuse ────────────────────────────────────────────────
            combined = fuse_results(scan_res, clin_res, symp_res, scan_label)

            # ══════════════════════════════════════════════════════
            # DISPLAY
            # ══════════════════════════════════════════════════════

            # ── Combined banner ─────────────────────────────────────
            combined_banner(
                combined['overall_label'],
                f"Urgency: {combined['urgency']} &nbsp;·&nbsp; "
                f"Follow-up: {combined['followup']}",
                combined['overall_confidence'],
                combined['color']
            )

            # Contribution bars
            contribution_bars(
                combined['scan_contribution'],
                combined['clinical_contribution'],
                combined['symptom_contribution']
            )

            # Three-way metric strip
            metric_grid([
                ("Overall Risk",    combined['overall_risk'],
                 get_risk_color(combined['overall_risk'])),
                ("Overall Conf.",   f"{combined['overall_confidence']}%",
                 get_risk_color(combined['overall_risk'])),
                ("Scan Model",      f"{scan_res['confidence']}%  {scan_res['risk_level']}",
                 '#1e293b'),
                ("Clinical Model",  f"{clin_res['confidence']}%  {clin_res['risk_level']}",
                 '#1e293b'),
                ("Symptoms",        f"{symp_res['label']}  ({symp_res['score']}/100)",
                 '#d97706' if symp_res['label'] in ('HIGH','MEDIUM') else '#15803d'),
                ("Model Agreement", "Yes" if clin_res['agreement'] else "No — review",
                 '#15803d' if clin_res['agreement'] else '#be123c'),
            ], cols=2)

            st.markdown("---")

            # ── Scan result section ─────────────────────────────────
            sh(f"Scan Analysis — {scan_label}")
            col_a, col_b = st.columns(2, gap="medium")
            with col_a:
                st.image(img_pil, caption="Original scan",
                         width="stretch")
            with col_b:
                banner(scan_res['label'],
                       f"{model_name} · {scan_res['risk_level']} risk · {elapsed}s",
                       get_risk_kind(scan_res['risk_level']))

                # Grad-CAM
                gradcam_path = None
                try:
                    from gradcam import generate_gradcam
                    cam_img, _ = generate_gradcam(img_pil, model, model_type)
                    st.image(cam_img, caption="Grad-CAM attention map",
                             width="stretch")
                    gcam_path_save = f"reports/gradcam_{scan_label.replace(' ','_')}_{pid}.png"
                    cam_img.save(gcam_path_save)
                    gradcam_path = gcam_path_save
                except Exception:
                    st.caption("Grad-CAM requires trained model weights.")

            if scan_res['risk_level'] in ('HIGH','MEDIUM'):
                interpretation = f"Imaging findings suggest elevated {scan_label} risk markers."
                action = "Immediate clinical correlation and physician referral."
            else:
                interpretation = f"No significant {scan_label} indicators on this scan."
                action = "Routine monitoring advised. Reassess if symptoms develop."

            scan_summary(f"{scan_label} — Detailed Imaging Report", [
                ("Scan modality",           scan_label,                              ""),
                ("AI model",                model_name,                              ""),
                ("Scan confidence",         f"{scan_res['confidence']}%",            ""),
                ("Scan risk level",         scan_res['risk_level'],
                 "danger" if scan_res['risk_level']=="HIGH" else
                 "warn"   if scan_res['risk_level']=="MEDIUM" else "safe"),
                ("Inference time",          f"{elapsed}s",                           ""),
                ("Interpretation",          interpretation,                          ""),
                ("Recommended action",      action,                                  ""),
            ])

            st.markdown("---")

            # ── Clinical result section ─────────────────────────────
            sh("Clinical Risk Assessment")
            raw = clinical_data['raw']

            col_c, col_d = st.columns(2, gap="medium")
            with col_c:
                banner(
                    f"Clinical Risk: {clin_res['risk_level']}",
                    f"XGBoost: {clin_res['xgb_prob']}% · "
                    f"Random Forest: {clin_res['rf_prob']}% · "
                    f"{clin_res.get('disagreement_note','')}",
                    get_risk_kind(clin_res['risk_level'])
                )
                scan_summary("Clinical Parameters", [
                    ("Age",          f"{raw.get('age','N/A')} years",
                     "danger" if raw.get('age',0) > 55 else ""),
                    ("Sex",           raw.get('sex','N/A'),                   ""),
                    ("BMI",          f"{raw.get('bmi','N/A')}",               ""),
                    ("Smoking",       raw.get('smoking','N/A'),               ""),
                    ("Diabetes",      raw.get('diabetes','N/A'),
                     "warn" if raw.get('diabetes') == 'Yes' else ""),
                    ("Blood pressure",f"{raw.get('trestbps','N/A')} mm Hg",
                     "danger" if raw.get('trestbps',0) > 140 else ""),
                    ("Cholesterol",  f"{raw.get('chol','N/A')} mg/dl",
                     "danger" if raw.get('chol',0) > 240 else ""),
                    ("Max heart rate",f"{raw.get('thalach','N/A')} bpm",     ""),
                    ("ST depression", f"{raw.get('oldpeak','N/A')}",
                     "warn" if raw.get('oldpeak',0) > 2 else ""),
                ])

            with col_d:
                # SHAP chart
                sh("SHAP Feature Importance")
                st.markdown(
                    '<p style="font-size:12px;color:#64748b;margin-bottom:10px">'
                    'Red = risk-increasing · Blue = risk-reducing</p>',
                    unsafe_allow_html=True)
                try:
                    explainer   = shap.Explainer(models['xgb'])
                    shap_vals   = explainer(clinical_data['encoded'])
                    shap_path   = f'reports/shap_{scan_label.replace(" ","_")}_{pid}.png'
                    fig, _      = plt.subplots(figsize=(7, 4))
                    shap.plots.waterfall(shap_vals[0], show=False)
                    plt.tight_layout()
                    plt.savefig(shap_path, dpi=130, bbox_inches='tight',
                                facecolor='#ffffff', edgecolor='none')
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
                except Exception as e:
                    st.warning(f"SHAP unavailable: {e}")
                    shap_path = None

            # Symptoms
            syms = raw.get('symptoms', [])
            if syms:
                st.markdown(
                    f'<div style="background:#fff;border:1px solid #e2e8f0;'
                    f'border-radius:10px;padding:12px 16px;margin-top:12px;font-size:12px">'
                    f'<strong style="color:#0f172a">Reported symptoms:</strong> '
                    f'<span style="color:#475569">{" · ".join(syms)}</span>'
                    f'<span style="float:right;color:#d97706;font-weight:600">'
                    f'Symptom severity: {symp_res["label"]} ({symp_res["score"]}/100)'
                    f'</span></div>',
                    unsafe_allow_html=True)

            st.markdown("---")

            # ── Final recommendations ───────────────────────────────
            sh("Final Recommendations")
            rec_list(combined['recommendations'])

            st.markdown("---")

            # ── PDF report ──────────────────────────────────────────
            sh("Download Combined Report")
            try:
                pdf_path = generate_combined_report(
                    patient_id      = pid,
                    physician       = phy,
                    scan_type       = scan_label,
                    clinical_notes  = notes or "",
                    scan_result     = scan_res,
                    clinical_result = clin_res,
                    symptom_result  = symp_res,
                    combined_result = combined,
                    raw_clinical    = raw,
                    shap_plot_path  = shap_path if 'shap_path' in dir() else None,
                    gradcam_path    = gradcam_path,
                )
                log_prediction(pid, scan_label, combined['overall_label'],
                               combined['overall_risk'],
                               combined['overall_confidence'], model_name, phy)
                log_id = log_audit(pid, scan_label, model_name,
                                   combined['overall_label'],
                                   combined['overall_risk'],
                                   combined['overall_confidence'],
                                   gradcam=gradcam_path is not None,
                                   report=True, physician=phy)

                dl_col, id_col = st.columns([2, 1])
                with dl_col:
                    with open(pdf_path, 'rb') as f:
                        st.download_button(
                            "Download Combined PDF Report",
                            data=f,
                            file_name=f"Hemo Check_{scan_label.replace(' ','_')}_{pid}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                with id_col:
                    st.markdown(
                        f'<p style="font-size:11px;color:#94a3b8;padding-top:10px">'
                        f'Audit ID: <code style="color:#0891b2">{log_id}</code></p>',
                        unsafe_allow_html=True)
            except Exception as e:
                st.error(f"PDF generation error: {e}")

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.info("Make sure all model .pth and .pkl files are in /models/")


# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
(tab_home, tab_ct, tab_mri, tab_us,
 tab_risk, tab_hist, tab_ana, tab_audit) = st.tabs([
    "Home",
    "CT Scan — Pulmonary Embolism",
    "MRI — Brain Clot",
    "Ultrasound — DVT",
    "Clinical Risk Only",
    "History",
    "Analytics",
    "Audit Trail",
])


# ══════════════════════════════════════════════════════════════════
# HOME TAB
# ══════════════════════════════════════════════════════════════════
with tab_home:
    st.markdown(f"""
    <div class="hero">
      <div style="position:relative;max-width:680px">
        <div class="hero-eyebrow">AI-Powered Medical Diagnostics</div>
        <div class="hero-title">Early <span>Blood Clot</span><br>Detection &amp; Risk<br>Assessment</div>
        <div class="hero-desc">
          Hemo Check is a multi-modal AI diagnostic system that combines deep learning image
          analysis with clinical risk prediction — delivering fast, explainable results with
          Grad-CAM heatmaps and SHAP feature importance.
        </div>
        <div class="hero-btns">
          <button class="hero-btn-p">Start Diagnosis</button>
          <button class="hero-btn-s">How It Works</button>
        </div>
        <div class="hero-stats">
          <div><div class="hs-num">4</div><div class="hs-lbl">DIAGNOSTIC MODULES</div></div>
          <div><div class="hs-num">3</div><div class="hs-lbl">DEEP LEARNING MODELS</div></div>
          <div><div class="hs-num">SHAP</div><div class="hs-lbl">AI EXPLAINABILITY</div></div>
          <div><div class="hs-num">Grad-CAM</div><div class="hs-lbl">VISUAL ATTENTION</div></div>
          <div><div class="hs-num">PDF</div><div class="hs-lbl">REPORT EXPORT</div></div>
        </div>
      </div>
    </div>

    <div class="stat-strip">
      <div class="stat-item"><div class="stat-num">~87%</div><div class="stat-lbl">CT MODEL ACCURACY</div></div>
      <div class="stat-item"><div class="stat-num">0.91</div><div class="stat-lbl">AUC SCORE (CLINICAL)</div></div>
      <div class="stat-item"><div class="stat-num">&lt;0.5s</div><div class="stat-lbl">INFERENCE TIME</div></div>
      <div class="stat-item"><div class="stat-num">100%</div><div class="stat-lbl">AUDIT LOGGED</div></div>
    </div>

    <div class="section-alt">
      <div style="padding:0 0 0 0">
        <div class="section-label">How It Works</div>
        <div class="section-title">Four diagnostic modules,<br>one unified platform</div>
        <div class="section-desc">Upload a scan, provide clinical details — Hemo Check runs both the imaging AI
        and the clinical risk model together, fusing results into one combined assessment.</div>
        <div class="module-grid">
          <div class="module-card">
            <div class="module-icon fi-blue"><i class="ti ti-lungs"></i></div>
            <div>
              <div class="module-title">CT Scan — Pulmonary Embolism</div>
              <div class="module-model">EfficientNet-B0 · RSNA 2020</div>
              <div class="module-desc">Detects blood clots in the pulmonary arteries from chest CT scans.
              Combined with clinical risk for a unified PE assessment.</div>
            </div>
          </div>
          <div class="module-card">
            <div class="module-icon fi-purple"><i class="ti ti-brain"></i></div>
            <div>
              <div class="module-title">MRI — Brain Clot and Stroke</div>
              <div class="module-model">ResNet-50 · Brain MRI Dataset</div>
              <div class="module-desc">Identifies brain clot indicators from MRI scans,
              fused with patient clinical data for a complete stroke risk picture.</div>
            </div>
          </div>
          <div class="module-card">
            <div class="module-icon fi-teal"><i class="ti ti-wave-sine"></i></div>
            <div>
              <div class="module-title">Ultrasound — DVT Detection</div>
              <div class="module-model">EfficientNet-B2 · Ultrasound Dataset</div>
              <div class="module-desc">Detects DVT markers in venous ultrasound images,
              combined with clinical features for integrated risk scoring.</div>
            </div>
          </div>
          <div class="module-card">
            <div class="module-icon fi-amber"><i class="ti ti-clipboard-list"></i></div>
            <div>
              <div class="module-title">Clinical Risk — Ensemble AI</div>
              <div class="module-model">XGBoost + Random Forest</div>
              <div class="module-desc">Standalone clinical risk predictor using 14 patient
              features. Also embedded in every scan tab for combined analysis.</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="section">
      <div class="section-label">Key Features</div>
      <div class="section-title">What makes Hemo Check faculty-grade</div>
      <div class="section-desc">Built to demonstrate end-to-end ML engineering,
      explainable AI, and full-stack deployment in a real healthcare context.</div>
      <div class="feat-grid">
        <div class="feat-card">
          <div class="feat-icon-wrap fi-teal"><i class="ti ti-eye"></i></div>
          <div class="feat-title">Grad-CAM Heatmaps</div>
          <div class="feat-desc">Visual attention maps overlaid on CT, MRI, and ultrasound images
          show exactly where the model focused to reach its decision.</div>
        </div>
        <div class="feat-card">
          <div class="feat-icon-wrap fi-blue"><i class="ti ti-chart-bar"></i></div>
          <div class="feat-title">SHAP Explainability</div>
          <div class="feat-desc">SHAP waterfall charts decompose every clinical risk prediction,
          showing which patient features increased or decreased the risk score.</div>
        </div>
        <div class="feat-card">
          <div class="feat-icon-wrap fi-purple"><i class="ti ti-git-merge"></i></div>
          <div class="feat-title">Fused AI Assessment</div>
          <div class="feat-desc">Scan model + clinical ensemble + symptom scorer combined
          with weighted fusion into one overall risk score per patient.</div>
        </div>
        <div class="feat-card">
          <div class="feat-icon-wrap fi-green"><i class="ti ti-shield-check"></i></div>
          <div class="feat-title">Confidence Thresholding</div>
          <div class="feat-desc">Predictions below 70% confidence are flagged as inconclusive
          rather than forced into a binary label — a critical safety feature.</div>
        </div>
        <div class="feat-card">
          <div class="feat-icon-wrap fi-amber"><i class="ti ti-history"></i></div>
          <div class="feat-title">Patient History</div>
          <div class="feat-desc">Every prediction stored in SQLite with patient ID, scan type,
          result, confidence, and physician. Risk trend charts per patient.</div>
        </div>
        <div class="feat-card">
          <div class="feat-icon-wrap fi-red"><i class="ti ti-file-export"></i></div>
          <div class="feat-title">PDF Report and Audit</div>
          <div class="feat-desc">Combined PDF reports include scan result, clinical assessment,
          SHAP chart, Grad-CAM, and recommendations in one document.</div>
        </div>
      </div>
    </div>

    <div style="background:#fff;border-top:1px solid #e2e8f0;padding:24px 40px">
      <div class="disclaimer">
        <div class="disc-icon"><i class="ti ti-info-circle"></i></div>
        <div>
          <div class="disc-title">Educational and Research Project</div>
          <div class="disc-body">Hemo Check is developed as a final year CSE project demonstrating
          AI/ML engineering skills. It is not a certified medical device and must not be used
          for real clinical diagnosis. Always consult a qualified healthcare professional.</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# CT SCAN TAB
# ══════════════════════════════════════════════════════════════════
with tab_ct:
    sh("CT Scan — Pulmonary Embolism Detection")
    st.markdown(
        '<p style="font-size:13px;color:#64748b;margin-bottom:20px">'
        'Upload a chest CT scan. After uploading, fill in the clinical details below. '
        'Hemo Check will run the EfficientNet-B0 scan model and the clinical ensemble together, '
        'then produce a combined AI assessment with Grad-CAM, SHAP, and a PDF report.</p>',
        unsafe_allow_html=True)

    ct_file = st.file_uploader(
        "Upload CT scan image (PNG / JPG / JPEG)",
        type=['png','jpg','jpeg'], key="ct")

    if ct_file:
        pt_info_block(pid, "CT Scan — PE", phy, today)
        st.markdown('<div class="info-bar"><strong>Step 2:</strong> Fill in the patient clinical '
                    'details below. Both the CT scan model and clinical risk model will run '
                    'together when you click Analyze.</div>', unsafe_allow_html=True)
        clinical_data = render_clinical_form("ct")
        st.markdown("---")
        if st.button("Analyze CT Scan + Clinical Risk", key="ct_run",
                     use_container_width=True):
            from ct_detector import predict_ct
            run_combined_scan_tab(
                ct_file, load_ct, predict_ct,
                'efficientnet', 'CT Scan', 'EfficientNet-B0',
                clinical_data, pid, phy, notes
            )
    else:
        st.markdown(
            '<p style="font-size:13px;color:#94a3b8;text-align:center;padding:40px 0">'
            'Upload a CT scan image to begin the combined analysis</p>',
            unsafe_allow_html=True)

    with st.expander("About pulmonary embolism and this model"):
        st.markdown("""<div style="font-size:13px;color:#475569;line-height:1.8">
        <strong style="color:#0f172a">Pulmonary Embolism (PE)</strong> is a life-threatening blockage
        in the pulmonary arteries caused by a blood clot.<br><br>
        <strong style="color:#0f172a">Model:</strong> EfficientNet-B0 with two-phase training.
        Phase 1 trains classifier head only; Phase 2 fine-tunes full network at LR x 0.1.<br><br>
        <strong style="color:#0f172a">Dataset:</strong> RSNA 2020 Pulmonary Embolism Detection Challenge.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# MRI TAB
# ══════════════════════════════════════════════════════════════════
with tab_mri:
    sh("MRI — Brain Clot and Stroke Indicator Detection")
    st.markdown(
        '<p style="font-size:13px;color:#64748b;margin-bottom:20px">'
        'Upload a brain MRI scan. ResNet-50 will analyze the image for clot indicators, '
        'combined with the clinical risk ensemble for a full brain clot risk assessment.</p>',
        unsafe_allow_html=True)

    mri_file = st.file_uploader(
        "Upload brain MRI image (PNG / JPG / JPEG)",
        type=['png','jpg','jpeg'], key="mri")

    if mri_file:
        pt_info_block(pid, "MRI — Brain Clot", phy, today)
        st.markdown('<div class="info-bar"><strong>Step 2:</strong> Fill in the clinical details. '
                    'Both the MRI scan model and clinical risk model will run together.</div>',
                    unsafe_allow_html=True)
        clinical_data = render_clinical_form("mri")
        st.markdown("---")
        if st.button("Analyze MRI + Clinical Risk", key="mri_run",
                     use_container_width=True):
            from mri_detector import predict_mri
            run_combined_scan_tab(
                mri_file, load_mri, predict_mri,
                'resnet', 'MRI Brain', 'ResNet-50',
                clinical_data, pid, phy, notes
            )
    else:
        st.markdown(
            '<p style="font-size:13px;color:#94a3b8;text-align:center;padding:40px 0">'
            'Upload an MRI scan image to begin the combined analysis</p>',
            unsafe_allow_html=True)

    with st.expander("About brain clots and this model"):
        st.markdown("""<div style="font-size:13px;color:#475569;line-height:1.8">
        <strong style="color:#0f172a">Brain clots</strong> cause ischemic strokes by blocking
        blood flow to brain tissue. Time-to-treatment is critical.<br><br>
        <strong style="color:#0f172a">Model:</strong> ResNet-50 fine-tuned on brain MRI dataset
        with custom classification head.<br><br>
        <strong style="color:#0f172a">Dataset:</strong> Brain MRI Images for Brain Tumour Detection (Kaggle).
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# ULTRASOUND TAB
# ══════════════════════════════════════════════════════════════════
with tab_us:
    sh("Ultrasound — Deep Vein Thrombosis Detection")
    st.markdown(
        '<p style="font-size:13px;color:#64748b;margin-bottom:20px">'
        'Upload an ultrasound image. EfficientNet-B2 analyzes venous patterns for DVT markers, '
        'combined with clinical data for a complete DVT risk assessment. '
        'Note: no public DVT-specific dataset exists — trained on closest proxy dataset.</p>',
        unsafe_allow_html=True)

    us_file = st.file_uploader(
        "Upload ultrasound image (PNG / JPG / JPEG)",
        type=['png','jpg','jpeg'], key="us")

    if us_file:
        pt_info_block(pid, "Ultrasound — DVT", phy, today)
        st.markdown('<div class="info-bar"><strong>Step 2:</strong> Fill in the clinical details. '
                    'Both the ultrasound model and clinical risk model will run together.</div>',
                    unsafe_allow_html=True)
        clinical_data = render_clinical_form("us")
        st.markdown("---")
        if st.button("Analyze Ultrasound + Clinical Risk", key="us_run",
                     use_container_width=True):
            from ultrasound_detector import predict_ultrasound
            run_combined_scan_tab(
                us_file, load_us, predict_ultrasound,
                'efficientnet', 'Ultrasound DVT', 'EfficientNet-B2',
                clinical_data, pid, phy, notes
            )
    else:
        st.markdown(
            '<p style="font-size:13px;color:#94a3b8;text-align:center;padding:40px 0">'
            'Upload an ultrasound image to begin the combined analysis</p>',
            unsafe_allow_html=True)

    with st.expander("About DVT and this model"):
        st.markdown("""<div style="font-size:13px;color:#475569;line-height:1.8">
        <strong style="color:#0f172a">DVT</strong> is a clot in a deep vein that can travel to
        the lungs causing pulmonary embolism.<br><br>
        <strong style="color:#0f172a">Dataset note:</strong> No labelled DVT ultrasound dataset
        is publicly available. This module uses the Breast Ultrasound dataset as a structural proxy.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# CLINICAL RISK ONLY TAB
# ══════════════════════════════════════════════════════════════════
with tab_risk:
    sh("Clinical Risk Assessment — Ensemble AI and SHAP")
    st.markdown(
        '<p style="font-size:13px;color:#64748b;margin-bottom:20px">'
        'Standalone clinical risk predictor — no scan image required. '
        'XGBoost and Random Forest ensemble with soft voting and SHAP explainability.</p>',
        unsafe_allow_html=True)

    clinical_data_r = render_clinical_form("risk")
    st.markdown("---")

    if st.button("Run Clinical Risk Assessment", use_container_width=True):
        clin_res = run_clinical_prediction(clinical_data_r['encoded'], models)
        symp_res = symptom_risk_score(clinical_data_r['raw'].get('symptoms', []))
        conf     = clin_res['confidence']
        rl       = clin_res['risk_level']
        rc       = get_risk_color(rl)

        banner(
            f"Clinical Risk: {rl}",
            f"Ensemble confidence: {conf}% · XGBoost: {clin_res['xgb_prob']}% · "
            f"RF: {clin_res['rf_prob']}% · {clin_res.get('disagreement_note','')}",
            get_risk_kind(rl)
        )

        metric_grid([
            ("Risk Level",    rl,               rc),
            ("Confidence",    f"{conf}%",        rc),
            ("XGBoost",       f"{clin_res['xgb_prob']}%", '#1e293b'),
            ("Random Forest", f"{clin_res['rf_prob']}%",  '#1e293b'),
        ])

        raw_r = clinical_data_r['raw']

        scan_summary("Clinical Parameter Summary", [
            ("Age",             f"{raw_r.get('age')} years",
             "danger" if raw_r.get('age',0) > 55 else ""),
            ("Blood pressure",  f"{raw_r.get('trestbps')} mm Hg",
             "danger" if raw_r.get('trestbps',0) > 140 else ""),
            ("Cholesterol",     f"{raw_r.get('chol')} mg/dl",
             "danger" if raw_r.get('chol',0) > 240 else ""),
            ("Max heart rate",  f"{raw_r.get('thalach')} bpm", ""),
            ("ST depression",   f"{raw_r.get('oldpeak')}",
             "warn" if raw_r.get('oldpeak',0) > 2 else ""),
            ("Symptom severity",f"{symp_res['label']} ({symp_res['score']}/100)",
             "danger" if symp_res['label'] == 'HIGH' else "warn" if symp_res['label'] == 'MEDIUM' else "safe"),
            ("Model agreement", "Yes" if clin_res['agreement'] else "No — review",
             "" if clin_res['agreement'] else "danger"),
        ])

        sh("SHAP Feature Importance")
        try:
            explainer   = shap.Explainer(models['xgb'])
            shap_vals   = explainer(clinical_data_r['encoded'])
            shap_path   = f'reports/shap_clinical_{pid}.png'
            fig, _      = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_vals[0], show=False)
            plt.tight_layout()
            plt.savefig(shap_path, dpi=150, bbox_inches='tight',
                        facecolor='#ffffff', edgecolor='none')
            st.pyplot(fig, use_container_width=True)
            plt.close()
        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")
            shap_path = None

        log_prediction(pid, 'Clinical Risk', f"Risk: {rl}", rl,
                       conf, 'XGBoost+RF Ensemble', phy)
        log_id = log_audit(pid, 'Clinical Risk', 'XGBoost+RF Ensemble',
                           f"Risk: {rl}", rl, conf, report=True, physician=phy)

        sh("Download Report")
        rpt = generate_report(
            patient_data=raw_r,
            prediction=int(clin_res['prediction']),
            confidence=conf,
            shap_plot_path=shap_path
        )
        dc, lc = st.columns(2)
        with dc:
            with open(rpt, 'rb') as f:
                st.download_button("Download PDF Report", data=f,
                    file_name="Hemo Check_Clinical_Report.pdf",
                    mime="application/pdf", use_container_width=True)
        with lc:
            st.markdown(
                f'<p style="font-size:11px;color:#94a3b8;padding-top:10px">'
                f'Audit: <code style="color:#0891b2">{log_id}</code></p>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# HISTORY TAB
# ══════════════════════════════════════════════════════════════════
with tab_hist:
    sh("Patient Prediction History")
    st.markdown(
        '<p style="font-size:13px;color:#64748b;margin-bottom:20px">'
        'All predictions stored in SQLite. Filter by patient ID.</p>',
        unsafe_allow_html=True)
    cf, cs = st.columns([3, 1])
    with cf:
        filter_pid = st.text_input("Filter by patient ID", value="", key="hf",
                                   placeholder="e.g. PT-2024-0001")
    with cs:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Refresh", key="hr", use_container_width=True):
            st.rerun()
    try:
        from history import get_patient_history
        df_h = (get_patient_history(filter_pid)
                if filter_pid else get_all_history(100))
        if df_h.empty:
            st.info("No predictions logged yet.")
        else:
            st.dataframe(df_h, use_container_width=True, hide_index=True)
            sh("Risk Distribution")
            rc_c = df_h['risk_level'].value_counts()
            cmap = {'HIGH':'#e11d48','MEDIUM':'#d97706','LOW':'#16a34a',
                    'VERY LOW':'#0891b2','INCONCLUSIVE':'#94a3b8'}
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.barh(rc_c.index, rc_c.values,
                    color=[cmap.get(r,'#94a3b8') for r in rc_c.index], height=0.5)
            for i, v in enumerate(rc_c.values):
                ax.text(v + 0.05, i, str(v), va='center', fontsize=10, color='#475569')
            ax.set_xlabel('Number of predictions', fontsize=10)
            ax.set_title('Risk Level Distribution', fontsize=12,
                         color='#0f172a', pad=10, fontweight='500')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
    except Exception as e:
        st.error(f"History unavailable: {e}")


# ══════════════════════════════════════════════════════════════════
# ANALYTICS TAB
# ══════════════════════════════════════════════════════════════════
with tab_ana:
    sh("Model Performance Analytics Dashboard")
    st.markdown(
        '<p style="font-size:13px;color:#64748b;margin-bottom:20px">'
        'Live performance metrics on held-out test data.</p>',
        unsafe_allow_html=True)
    if st.button("Compute Analytics", use_container_width=True):
        try:
            X_test = pd.read_csv('data/X_test.csv')
            y_test = pd.read_csv('data/y_test.csv').squeeze()
            xm     = compute_metrics(models['xgb'], X_test, y_test)
            rm     = compute_metrics(models['rf'],  X_test, y_test)
            c1, c2 = st.columns(2, gap="large")
            with c1:
                sh("XGBoost")
                metric_grid([
                    ("Accuracy",  f"{xm['accuracy']}%",  '#0891b2'),
                    ("AUC",       f"{xm['auc']}%",        '#16a34a'),
                    ("Precision", f"{xm['precision']}%",  '#1e293b'),
                    ("Recall",    f"{xm['recall']}%",     '#1e293b'),
                ])
            with c2:
                sh("Random Forest")
                metric_grid([
                    ("Accuracy",  f"{rm['accuracy']}%",  '#0891b2'),
                    ("AUC",       f"{rm['auc']}%",        '#16a34a'),
                    ("Precision", f"{rm['precision']}%",  '#1e293b'),
                    ("Recall",    f"{rm['recall']}%",     '#1e293b'),
                ])
            sh("ROC Curve Comparison")
            roc_p = plot_roc_comparison(
                {'XGBoost':models['xgb'],'Random Forest':models['rf']},
                X_test, y_test)
            st.image(roc_p, width="stretch")
            ca2, cb2 = st.columns(2)
            with ca2:
                sh("Confusion Matrix")
                cm_p = plot_confusion_matrix(xm['cm'], 'XGBoost')
                st.image(cm_p, width="stretch")
            with cb2:
                sh("Feature Importance")
                fi_p = plot_feature_importance(models['xgb'], list(X_test.columns))
                st.image(fi_p, width="stretch")
            sh("Precision-Recall Curve")
            pr_p = plot_precision_recall(models['xgb'], X_test, y_test)
            st.image(pr_p, width="stretch")
        except FileNotFoundError:
            st.warning("Run preprocess.py and train_model.py first.")
        except Exception as e:
            st.error(f"Analytics error: {e}")
    else:
        st.markdown(
            '<p style="font-size:13px;color:#94a3b8;text-align:center;padding:40px 0">'
            'Click Compute Analytics to load performance charts</p>',
            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# AUDIT TRAIL TAB
# ══════════════════════════════════════════════════════════════════
with tab_audit:
    sh("System Audit Trail and Prediction Logs")
    st.markdown(
        '<p style="font-size:13px;color:#64748b;margin-bottom:20px">'
        'Every prediction logged with timestamp, model version, '
        'Grad-CAM status, and report generation. Exportable as CSV.</p>',
        unsafe_allow_html=True)
    ar, ae = st.columns([3, 1])
    with ar:
        if st.button("Refresh Audit Log", key="arf", use_container_width=True):
            st.rerun()
    with ae:
        try:
            csv_p = export_audit_csv()
            with open(csv_p, 'rb') as f:
                st.download_button("Export CSV", data=f,
                    file_name="Hemo Check_audit_log.csv",
                    mime="text/csv", use_container_width=True)
        except Exception:
            pass
    try:
        df_a = get_audit_log(200)
        if df_a.empty:
            st.info("No audit records yet.")
        else:
            st.dataframe(df_a, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Audit log unavailable: {e}")


# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="site-footer">
  <div style="display:flex;align-items:center;gap:12px">
    <img src="data:image/png;base64,{logo}"
         style="width:48px;height:48px;border-radius:10px;">
    <div>
      <div class="sf-brand">Hemo Check</div>
      <div class="sf-tag">Blood Clot Diagnostic Intelligence System · v3.0</div>
    </div>
  </div>

  <div class="sf-copy">
    For educational and research purposes only &nbsp;|&nbsp;
    Not for clinical diagnosis &nbsp;|&nbsp;
  </div>
</div>
""", unsafe_allow_html=True)