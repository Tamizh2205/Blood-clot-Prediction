"""
report.py
---------
Generates PDF medical reports using fpdf2 with DejaVu Unicode font
to support all special characters. Covers:
  1. Patient demographics
  2. Scan analysis result
  3. Clinical risk assessment
  4. Combined AI assessment
  5. Recommendations
  6. SHAP + Grad-CAM images
"""

from fpdf import FPDF
from datetime import datetime
import os, urllib.request, re

os.makedirs('reports', exist_ok=True)
os.makedirs('fonts',   exist_ok=True)

# ── Download DejaVu font if missing (Unicode support) ──────────────
FONT_URL  = ("https://github.com/dejavu-fonts/dejavu-fonts/raw/master/"
             "ttf/DejaVuSans.ttf")
FONT_B_URL= ("https://github.com/dejavu-fonts/dejavu-fonts/raw/master/"
             "ttf/DejaVuSans-Bold.ttf")
FONT_I_URL= ("https://github.com/dejavu-fonts/dejavu-fonts/raw/master/"
             "ttf/DejaVuSans-Oblique.ttf")

def _dl(url, path):
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(url, path)
        except Exception:
            pass

_dl(FONT_URL,   'fonts/DejaVuSans.ttf')
_dl(FONT_B_URL, 'fonts/DejaVuSans-Bold.ttf')
_dl(FONT_I_URL, 'fonts/DejaVuSans-Oblique.ttf')

HAVE_DEJAVU = os.path.exists('fonts/DejaVuSans.ttf')

# ── Sanitise any remaining non-latin-1 chars as safety net ─────────
def _s(text: str) -> str:
    """Sanitise string — remove/replace chars outside latin-1 range."""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        '\u2014': '-', '\u2013': '-', '\u00b7': '.', '\u2022': '-',
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
        '\u2026': '...', '\u2500': '-', '\u2550': '=', '\u2502': '|',
        '\u2503': '|', '\u25cf': '-', '\u2713': 'OK', '\u2717': 'X',
        '\u00d7': 'x', '\u2265': '>=', '\u2264': '<=', '\u00b0': 'deg',
        '\u00ae': '(R)', '\u00a9': '(C)', '\u2122': '(TM)',
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Final fallback: drop anything still outside latin-1
    return text.encode('latin-1', errors='replace').decode('latin-1')

# ── Colour palette ─────────────────────────────────────────────────
NAVY   = (15,  23,  42)
TEAL   = ( 8, 145, 178)
WHITE  = (255, 255, 255)
LIGHT  = (248, 250, 252)
BORDER = (226, 232, 240)
RED    = (190,  18,  60)
GREEN  = ( 21, 128,  61)
AMBER  = (146,  64,  14)
MUTED  = (100, 116, 139)
TEXT   = ( 30,  41,  59)


class HemoCheckReport(FPDF):

    def __init__(self):
        super().__init__()
        if HAVE_DEJAVU:
            self.add_font('DV',  '',  'fonts/DejaVuSans.ttf',        uni=True)
            self.add_font('DV',  'B', 'fonts/DejaVuSans-Bold.ttf',   uni=True)
            self.add_font('DV',  'I', 'fonts/DejaVuSans-Oblique.ttf',uni=True)
            self._fn = 'DV'
        else:
            self._fn = 'Helvetica'

    def _font(self, style='', size=10):
        self.set_font(self._fn, style, size)

    def header(self):
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 20, 'F')
        logo_path = "assets/favicon.png"
        if os.path.exists(logo_path):
            self.image(logo_path, x=10, y=3, w=14)
        self._font('B', 13)
        self.set_text_color(*WHITE)
        self.set_xy(28, 5)
        self.cell(80, 6, 'Hemo Check', ln=0)
        self._font('', 8)
        self.set_text_color(186, 230, 253)
        self.set_xy(26, 12)
        self.cell(80, 5, 'Diagnostic Intelligence System  |  v3.0', ln=0)
        self._font('', 8)
        self.set_text_color(186, 230, 253)
        self.set_xy(130, 8)
        self.cell(70, 6,
                  _s(f'Generated: {datetime.now().strftime("%d %b %Y  %H:%M")}'),
                  align='R')
        self.set_text_color(*TEXT)
        self.ln(22)

    def footer(self):
        self.set_y(-14)
        self._font('I', 7)
        self.set_text_color(*MUTED)
        self.cell(0, 5,
                  _s('FOR EDUCATIONAL PURPOSES ONLY  |  NOT FOR CLINICAL DIAGNOSIS'
                     f'  |  Page {self.page_no()}'),
                  align='C')

    def section_title(self, title: str):
        self.set_fill_color(*TEAL)
        self.rect(10, self.get_y(), 190, 8, 'F')
        self._font('B', 9)
        self.set_text_color(*WHITE)
        self.set_x(13)
        self.cell(0, 8, _s(f'  {title.upper()}'), ln=True)
        self.set_text_color(*TEXT)
        self.ln(2)

    def key_value(self, label: str, value: str, value_color=None):
        self._font('', 9)
        self.set_text_color(*MUTED)
        self.set_x(12)
        self.cell(68, 7, _s(label + ':'), border='B')
        self._font('B', 9)
        self.set_text_color(*(value_color or TEXT))
        self.cell(0, 7, _s(str(value)), border='B', ln=True)
        self.set_text_color(*TEXT)

    def risk_banner(self, label: str, confidence, risk_level: str):
        colors = {
            'HIGH'        : ((255, 241, 242), RED),
            'MEDIUM'      : ((255, 251, 235), AMBER),
            'LOW'         : ((240, 253, 244), GREEN),
            'VERY LOW'    : ((236, 254, 255), TEAL),
            'INCONCLUSIVE': ((248, 250, 252), MUTED),
        }
        fill, text_c = colors.get(risk_level, ((248, 250, 252), MUTED))
        self.set_fill_color(*fill)
        self.set_draw_color(*text_c)
        self.set_line_width(0.8)
        self.rect(12, self.get_y(), 186, 14, 'DF')
        self.set_line_width(0.2)
        self.set_draw_color(*BORDER)
        self._font('B', 11)
        self.set_text_color(*text_c)
        self.set_x(16)
        self.cell(140, 14, _s(label), ln=False)
        self._font('', 9)
        self.cell(0, 14, _s(f'{confidence}%  confidence'), align='R', ln=True)
        self.set_text_color(*TEXT)
        self.ln(3)

    def two_col(self, left_pairs, right_pairs):
        y_start = self.get_y()
        def draw_col(pairs, x):
            y = y_start
            for label, value, color in pairs:
                self.set_xy(x, y)
                self._font('', 8)
                self.set_text_color(*MUTED)
                self.cell(45, 6, _s(label + ':'), border='B')
                self._font('B', 8)
                self.set_text_color(*(color or TEXT))
                self.cell(45, 6, _s(str(value)), border='B')
                y += 6
        draw_col(left_pairs,  12)
        draw_col(right_pairs, 112)
        self.set_text_color(*TEXT)
        rows = max(len(left_pairs), len(right_pairs))
        self.set_y(y_start + rows * 6 + 4)

    def rec_item(self, index: int, text: str):
        self._font('B', 9)
        self.set_text_color(*TEAL)
        self.set_x(12)
        self.cell(8, 7, _s(f'{index}.'))
        self._font('', 9)
        self.set_text_color(*TEXT)
        self.multi_cell(176, 5, _s(text))
        self.ln(1)

    def divider(self):
        self.set_draw_color(*BORDER)
        self.set_line_width(0.3)
        self.line(12, self.get_y(), 198, self.get_y())
        self.ln(3)


# ══════════════════════════════════════════════════════════════════
# COMBINED REPORT
# ══════════════════════════════════════════════════════════════════
def generate_combined_report(
    patient_id     : str,
    physician      : str,
    scan_type      : str,
    clinical_notes : str,
    scan_result    : dict,
    clinical_result: dict,
    symptom_result : dict,
    combined_result: dict,
    raw_clinical   : dict,
    shap_plot_path : str = None,
    gradcam_path   : str = None,
) -> str:

    pdf = HemoCheckReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(10, 10, 10)

    report_id = f'Hemo Check-{datetime.now().strftime("%Y%m%d%H%M%S")}'
    rl_color  = {
        'HIGH': RED, 'MEDIUM': AMBER,
        'LOW': GREEN, 'VERY LOW': TEAL, 'INCONCLUSIVE': MUTED,
    }

    # metadata strip
    pdf._font('', 8)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 5,
             _s(f'Report ID: {report_id}  |  Patient: {patient_id}'
                f'  |  Physician: {physician}  |  Scan: {scan_type}'),
             ln=True)
    pdf.ln(3)

    # ── Section 1: Combined banner ─────────────────────────────────
    pdf.section_title('Combined AI Assessment')
    pdf.risk_banner(
        combined_result['overall_label'],
        combined_result['overall_confidence'],
        combined_result['overall_risk']
    )

    pdf._font('', 8)
    pdf.set_text_color(*MUTED)
    pdf.set_x(12)
    pdf.cell(0, 5,
             _s(f"Scan model: {combined_result['scan_contribution']}%  |  "
                f"Clinical model: {combined_result['clinical_contribution']}%  |  "
                f"Symptom score: {combined_result['symptom_contribution']}%"),
             ln=True)
    pdf.set_text_color(*TEXT)
    pdf.ln(2)

    for label, value, kind in combined_result['summary_lines']:
        color = (RED   if kind == 'danger' else
                 AMBER if kind == 'warn'   else
                 GREEN if kind == 'safe'   else TEXT)
        pdf.key_value(label, value, value_color=color)
    pdf.ln(4)

    # ── Section 2: Scan result ─────────────────────────────────────
    pdf.section_title(f'{scan_type} - Imaging Analysis')
    scan_rl = scan_result.get('risk_level', 'N/A')
    pdf.risk_banner(
        scan_result.get('label', 'N/A'),
        scan_result.get('confidence', 0),
        scan_rl
    )
    pdf.two_col(
        left_pairs=[
            ('Scan type',  scan_type,                                 None),
            ('Model',      scan_result.get('model_name', 'AI Model'), None),
            ('Risk level', scan_rl, rl_color.get(scan_rl, TEXT)),
        ],
        right_pairs=[
            ('Confidence', f"{scan_result.get('confidence', 0)}%",   None),
            ('Inference',  scan_result.get('elapsed', 'N/A'),         None),
            ('Grad-CAM',   'Generated' if gradcam_path else 'N/A',    None),
        ]
    )

    if gradcam_path and os.path.exists(gradcam_path):
        pdf.ln(2)
        pdf._font('B', 8)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 5, 'Grad-CAM Attention Heatmap', ln=True)
        try:
            pdf.image(gradcam_path, x=12, w=90)
        except Exception:
            pass
        pdf.ln(2)
    pdf.ln(3)

    # ── Section 3: Clinical assessment ────────────────────────────
    pdf.section_title('Clinical Risk Assessment')
    clin_rl = clinical_result.get('risk_level', 'N/A')
    pdf.risk_banner(
        f"Clinical Risk: {clin_rl}",
        clinical_result.get('confidence', 0),
        clin_rl
    )
    pdf.two_col(
        left_pairs=[
            ('Age',      f"{raw_clinical.get('age', 'N/A')} years",  None),
            ('Sex',       raw_clinical.get('sex', 'N/A'),             None),
            ('BMI',      f"{raw_clinical.get('bmi', 'N/A')}",        None),
            ('Smoking',   raw_clinical.get('smoking', 'N/A'),         None),
            ('Diabetes',  raw_clinical.get('diabetes', 'N/A'),        None),
        ],
        right_pairs=[
            ('Blood pressure',
             f"{raw_clinical.get('trestbps', 'N/A')} mm Hg",
             RED if raw_clinical.get('trestbps', 0) > 140 else None),
            ('Cholesterol',
             f"{raw_clinical.get('chol', 'N/A')} mg/dl",
             RED if raw_clinical.get('chol', 0) > 240 else None),
            ('Max heart rate',
             f"{raw_clinical.get('thalach', 'N/A')} bpm", None),
            ('ST depression',
             f"{raw_clinical.get('oldpeak', 'N/A')}",     None),
            ('XGBoost prob.',
             f"{clinical_result.get('xgb_prob', 'N/A')}%",None),
        ]
    )

    symptoms = raw_clinical.get('symptoms', [])
    if symptoms:
        pdf.ln(2)
        pdf._font('B', 8)
        pdf.set_text_color(*MUTED)
        pdf.set_x(12)
        pdf.cell(0, 5, 'Reported Symptoms:', ln=True)
        pdf._font('', 8)
        pdf.set_text_color(*TEXT)
        pdf.set_x(12)
        pdf.multi_cell(0, 5, _s('  - ' + '\n  - '.join(symptoms)))

    pdf.ln(2)
    symp_label = symptom_result.get('label', 'NONE')
    symp_score = symptom_result.get('score', 0)
    pdf.key_value(
        'Symptom severity score',
        f'{symp_label} ({symp_score}/100)',
        value_color=(RED if symp_label == 'HIGH' else
                     AMBER if symp_label == 'MEDIUM' else GREEN)
    )

    if not clinical_result.get('agreement', True):
        pdf.ln(2)
        pdf._font('I', 8)
        pdf.set_text_color(*AMBER)
        pdf.set_x(12)
        pdf.multi_cell(0, 5,
            _s(f"Note: {clinical_result.get('disagreement_note', '')}"))
    pdf.ln(3)

    # ── Section 4: SHAP ───────────────────────────────────────────
    if shap_plot_path and os.path.exists(shap_plot_path):
        pdf.section_title('AI Explainability - SHAP Feature Analysis')
        pdf._font('', 8)
        pdf.set_text_color(*MUTED)
        pdf.set_x(12)
        pdf.multi_cell(0, 5,
            _s('Red bars increase clot risk; blue bars reduce risk. '
               'Bar length = magnitude of impact on the prediction.'))
        pdf.ln(2)
        try:
            pdf.image(shap_plot_path, x=12, w=180)
        except Exception:
            pass
        pdf.ln(3)

    # ── Section 5: Recommendations ────────────────────────────────
    pdf.section_title('Clinical Recommendations')
    for i, rec in enumerate(combined_result.get('recommendations', []), 1):
        pdf.rec_item(i, rec)

    pdf.ln(2)
    pdf.key_value('Urgency',     combined_result.get('urgency', 'N/A'))
    pdf.key_value('Follow-up',   combined_result.get('followup', 'N/A'))

    if clinical_notes and clinical_notes.strip():
        pdf.ln(3)
        pdf.section_title('Physician Notes')
        pdf._font('', 9)
        pdf.set_text_color(*TEXT)
        pdf.set_x(12)
        pdf.multi_cell(0, 6, _s(clinical_notes.strip()))

    # ── Disclaimer ────────────────────────────────────────────────
    pdf.ln(4)
    pdf.set_fill_color(*LIGHT)
    pdf.set_draw_color(*BORDER)
    pdf.rect(10, pdf.get_y(), 190, 18, 'DF')
    pdf._font('B', 8)
    pdf.set_text_color(*MUTED)
    pdf.set_x(14)
    pdf.cell(0, 6, 'DISCLAIMER', ln=True)
    pdf._font('I', 8)
    pdf.set_x(14)
    pdf.multi_cell(182, 5,
        _s('This report is generated by an AI system for educational and research '
           'purposes only. It does not constitute medical advice and must not be '
           'used for clinical diagnosis or treatment decisions. Always consult a '
           'qualified healthcare professional for medical evaluation.'))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename  = (f'reports/Hemo Check_{scan_type.replace(" ", "_")}'
                 f'_{patient_id}_{timestamp}.pdf')
    pdf.output(filename)
    print(f'Report saved: {filename}')
    return filename


# ── Legacy wrapper ────────────────────────────────────────────────
def generate_report(patient_data: dict, prediction: int,
                    confidence: float,
                    shap_plot_path: str = None) -> str:
    pdf = HemoCheckReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=18)
    rl = 'HIGH' if prediction == 1 else 'LOW'
    pdf.section_title('Clinical Risk Assessment')
    pdf.risk_banner(
        'High Clot Risk Detected' if prediction == 1 else 'Low Clot Risk',
        confidence, rl
    )
    for key, value in patient_data.items():
        pdf.key_value(key.replace('_', ' ').title(), str(value))
    if shap_plot_path and os.path.exists(shap_plot_path):
        pdf.ln(4)
        pdf.section_title('SHAP Explainability')
        try:
            pdf.image(shap_plot_path, x=12, w=180)
        except Exception:
            pass
    pdf.ln(4)
    pdf._font('I', 8)
    pdf.set_text_color(*MUTED)
    pdf.set_x(12)
    pdf.multi_cell(0, 5,
        _s('This report is for educational purposes only. '
           'Not a substitute for medical advice.'))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename  = f'reports/Hemo Check_Clinical_{timestamp}.pdf'
    pdf.output(filename)
    return filename