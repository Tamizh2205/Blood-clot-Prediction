"""
report.py
─────────
Generates a comprehensive PDF medical report covering:
  1. Patient demographics
  2. Scan analysis result (CT / MRI / Ultrasound)
  3. Clinical risk assessment
  4. Combined AI assessment
  5. Recommendations
  6. SHAP explanation (optional image)
"""

from fpdf import FPDF
from datetime import datetime
import os

os.makedirs('reports', exist_ok=True)

# ── Brand colours ──────────────────────────────────────────────────
NAVY    = (15,  23,  42)
TEAL    = ( 8, 145, 178)
WHITE   = (255, 255, 255)
LIGHT   = (248, 250, 252)
BORDER  = (226, 232, 240)
RED     = (190,  18,  60)
GREEN   = ( 21, 128,  61)
AMBER   = (146,  64,  14)
MUTED   = (100, 116, 139)
TEXT    = ( 30,  41,  59)


class HemoCheckReport(FPDF):

    def header(self):
        # Navy header bar
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 20, 'F')

        # Logo placeholder circle
        self.set_fill_color(*TEAL)
        self.ellipse(10, 4, 12, 12, 'F')

        # System name
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(*WHITE)
        self.set_xy(26, 6)
        self.cell(80, 6, 'Hemo Check', ln=0)

        self.set_font('Helvetica', '', 8)
        self.set_text_color(186, 230, 253)
        self.set_xy(26, 12)
        self.cell(80, 5, 'Diagnostic Intelligence System  |  v3.0', ln=0)

        # Date right
        self.set_font('Helvetica', '', 8)
        self.set_text_color(186, 230, 253)
        self.set_xy(130, 8)
        self.cell(70, 6,
                  f'Generated: {datetime.now().strftime("%d %b %Y  %H:%M")}',
                  align='R')
        self.set_text_color(*TEXT)
        self.ln(22)

    def footer(self):
        self.set_y(-14)
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(*MUTED)
        self.cell(0, 5,
                  'FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY  |  '
                  'NOT FOR CLINICAL DIAGNOSIS  |  '
                  f'Page {self.page_no()}',
                  align='C')

    def section_title(self, title: str, icon: str = ''):
        self.set_fill_color(*TEAL)
        self.rect(10, self.get_y(), 190, 8, 'F')
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(*WHITE)
        self.set_x(13)
        self.cell(0, 8, f'  {icon}  {title.upper()}' if icon else f'  {title.upper()}',
                  ln=True)
        self.set_text_color(*TEXT)
        self.ln(2)

    def key_value(self, label: str, value: str,
                  value_color=None, bold_val=False):
        self.set_font('Helvetica', '', 9)
        self.set_text_color(*MUTED)
        self.set_x(12)
        self.cell(65, 7, label + ':', border='B',
                  fill=False)
        self.set_font('Helvetica', 'B' if bold_val else '', 9)
        self.set_text_color(*(value_color or TEXT))
        self.cell(0, 7, str(value), border='B', ln=True)
        self.set_text_color(*TEXT)

    def two_col(self, left_pairs, right_pairs):
        y_start = self.get_y()
        x_left  = 12
        x_right = 112

        def draw_col(pairs, x):
            y = y_start
            for label, value, color in pairs:
                self.set_xy(x, y)
                self.set_font('Helvetica', '', 8)
                self.set_text_color(*MUTED)
                self.cell(45, 6, label + ':', border='B')
                self.set_font('Helvetica', 'B', 8)
                self.set_text_color(*(color or TEXT))
                self.cell(45, 6, str(value), border='B', ln=False)
                y += 6

        draw_col(left_pairs,  x_left)
        draw_col(right_pairs, x_right)
        self.set_text_color(*TEXT)
        # advance to below both columns
        rows = max(len(left_pairs), len(right_pairs))
        self.set_y(y_start + rows * 6 + 4)

    def risk_banner(self, label: str, confidence: float,
                    risk_level: str):
        if risk_level == 'HIGH':
            fill, text_c = (255, 241, 242), RED
        elif risk_level == 'MEDIUM':
            fill, text_c = (255, 251, 235), AMBER
        elif risk_level in ('LOW', 'VERY LOW'):
            fill, text_c = (240, 253, 244), GREEN
        else:
            fill, text_c = (248, 250, 252), MUTED

        self.set_fill_color(*fill)
        self.set_draw_color(*text_c)
        self.set_line_width(0.8)
        self.rect(12, self.get_y(), 186, 14, 'DF')
        self.set_line_width(0.2)
        self.set_draw_color(*BORDER)

        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(*text_c)
        self.set_x(16)
        self.cell(140, 14, label, ln=False)
        self.set_font('Helvetica', '', 9)
        self.cell(0, 14, f'{confidence}%  confidence', align='R', ln=True)
        self.set_text_color(*TEXT)
        self.ln(3)

    def recommendation_item(self, index: int, text: str):
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(*TEAL)
        self.set_x(12)
        self.cell(8, 7, f'{index}.')
        self.set_font('Helvetica', '', 9)
        self.set_text_color(*TEXT)
        self.multi_cell(176, 5, text)
        self.ln(1)


# �-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-�
# MAIN GENERATE FUNCTION
# �-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-�
def generate_combined_report(
    patient_id      : str,
    physician       : str,
    scan_type       : str,
    clinical_notes  : str,
    scan_result     : dict,
    clinical_result : dict,
    symptom_result  : dict,
    combined_result : dict,
    raw_clinical    : dict,
    shap_plot_path  : str = None,
    gradcam_path    : str = None,
) -> str:
    """
    Generate the full combined PDF report.

    Parameters
    ----------
    patient_id      : e.g. "PT-2024-0001"
    physician       : e.g. "Dr. Sharma"
    scan_type       : "CT Scan" | "MRI Brain" | "Ultrasound DVT"
    clinical_notes  : free-text notes from sidebar
    scan_result     : dict from image detector (label, confidence, risk_level, raw_prob)
    clinical_result : dict from ensemble_predict()
    symptom_result  : dict from symptom_risk_score()
    combined_result : dict from fuse_results()
    raw_clinical    : raw display values from clinical panel
    shap_plot_path  : path to SHAP waterfall image
    gradcam_path    : path to Grad-CAM overlay image

    Returns
    -------
    str — file path of the saved PDF
    """

    pdf = HemoCheckReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(10, 10, 10)

    report_id = f"Hemo Check-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    rl_color  = {
        'HIGH'        : RED,
        'MEDIUM'      : AMBER,
        'LOW'         : GREEN,
        'VERY LOW'    : (8, 145, 178),
        'INCONCLUSIVE': MUTED,
    }

    # ── Report metadata strip ──────────────────────────────────────
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 5,
             f'Report ID: {report_id}   |   '
             f'Patient ID: {patient_id}   |   '
             f'Physician: {physician}   |   '
             f'Scan: {scan_type}',
             ln=True)
    pdf.ln(3)

    # �-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-�
    # SECTION 1 — COMBINED RESULT BANNER
    # �-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-�
    pdf.section_title('Combined AI Assessment')
    pdf.risk_banner(
        combined_result['overall_label'],
        combined_result['overall_confidence'],
        combined_result['overall_risk']
    )

    # Three-way contribution bar
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(*MUTED)
    pdf.set_x(12)
    pdf.cell(0, 5,
             f"Scan model: {combined_result['scan_contribution']}%  |  "
             f"Clinical model: {combined_result['clinical_contribution']}%  |  "
             f"Symptom score: {combined_result['symptom_contribution']}%",
             ln=True)
    pdf.set_text_color(*TEXT)
    pdf.ln(3)

    # Summary table
    left_summary  = combined_result['summary_lines'][:6]
    right_summary = combined_result['summary_lines'][6:]
    for label, value, _ in left_summary + right_summary:
        color = (RED if _ == 'danger' else
                 AMBER if _ == 'warn' else
                 GREEN if _ == 'safe' else TEXT)
        pdf.key_value(label, value, value_color=color)
    pdf.ln(4)

    # �-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-�
    # SECTION 2 — SCAN RESULT
    # �-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-�
    pdf.section_title(f'{scan_type} — Imaging Analysis')
    scan_rl = scan_result.get('risk_level', 'N/A')
    scan_c  = rl_color.get(scan_rl, TEXT)
    pdf.risk_banner(
        scan_result.get('label', 'N/A'),
        scan_result.get('confidence', 0),
        scan_rl
    )
    pdf.two_col(
        left_pairs=[
            ('Scan type',    scan_type,                              None),
            ('Model',        scan_result.get('model_name','AI Model'),None),
            ('Risk level',   scan_rl,                               scan_c),
        ],
        right_pairs=[
            ('Confidence',   f"{scan_result.get('confidence',0)}%", None),
            ('Inference',    scan_result.get('elapsed','N/A'),      None),
            ('Grad-CAM',     'Generated' if gradcam_path else 'N/A',None),
        ]
    )

    # Grad-CAM image
    if gradcam_path and os.path.exists(gradcam_path):
        pdf.ln(2)
        pdf.set_font('Helvetica', 'B', 8)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 5, 'Grad-CAM Attention Heatmap', ln=True)
        try:
            pdf.image(gradcam_path, x=12, w=90)
        except Exception:
            pass
        pdf.ln(2)

    pdf.ln(3)

    # �-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-�
    # SECTION 3 — CLINICAL ASSESSMENT
    # �-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-�
    pdf.section_title('Clinical Risk Assessment')
    clin_rl = clinical_result.get('risk_level', 'N/A')
    clin_c  = rl_color.get(clin_rl, TEXT)
    pdf.risk_banner(
        f"Clinical Risk: {clin_rl}",
        clinical_result.get('confidence', 0),
        clin_rl
    )

    pdf.two_col(
        left_pairs=[
            ('Age',            f"{raw_clinical.get('age','N/A')} years",  None),
            ('Sex',             raw_clinical.get('sex','N/A'),            None),
            ('BMI',            f"{raw_clinical.get('bmi','N/A')}",        None),
            ('Smoking',         raw_clinical.get('smoking','N/A'),        None),
            ('Diabetes',        raw_clinical.get('diabetes','N/A'),       None),
        ],
        right_pairs=[
            ('Blood pressure', f"{raw_clinical.get('trestbps','N/A')} mm Hg",
             RED if raw_clinical.get('trestbps',0) > 140 else None),
            ('Cholesterol',    f"{raw_clinical.get('chol','N/A')} mg/dl",
             RED if raw_clinical.get('chol',0) > 240 else None),
            ('Max heart rate', f"{raw_clinical.get('thalach','N/A')} bpm", None),
            ('ST depression',  f"{raw_clinical.get('oldpeak','N/A')}",    None),
            ('XGBoost prob.',  f"{clinical_result.get('xgb_prob','N/A')}%",None),
        ]
    )

    # Symptoms
    symptoms = raw_clinical.get('symptoms', [])
    if symptoms:
        pdf.ln(2)
        pdf.set_font('Helvetica', 'B', 8)
        pdf.set_text_color(*MUTED)
        pdf.set_x(12)
        pdf.cell(0, 5, 'Reported Symptoms:', ln=True)
        pdf.set_font('Helvetica', '', 8)
        pdf.set_text_color(*TEXT)
        pdf.set_x(12)
        pdf.multi_cell(0, 5, '  •  ' + '\n  •  '.join(symptoms))

    # Symptom risk score
    pdf.ln(2)
    symp_label = symptom_result.get('label', 'NONE')
    symp_score = symptom_result.get('score', 0)
    pdf.key_value('Symptom severity score',
                  f"{symp_label} ({symp_score}/100)",
                  value_color=RED if symp_label == 'HIGH' else AMBER if symp_label == 'MEDIUM' else GREEN)

    # Model agreement note
    if not clinical_result.get('agreement', True):
        pdf.ln(2)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(*AMBER)
        pdf.set_x(12)
        pdf.multi_cell(0, 5,
            f"Note: {clinical_result.get('disagreement_note','')}")
    pdf.ln(3)

    # �-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-�
    # SECTION 4 — SHAP EXPLANATION
    # �-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-�
    if shap_plot_path and os.path.exists(shap_plot_path):
        pdf.section_title('AI Explainability — SHAP Feature Analysis')
        pdf.set_font('Helvetica', '', 8)
        pdf.set_text_color(*MUTED)
        pdf.set_x(12)
        pdf.multi_cell(0, 5,
            'The chart below shows which clinical features most influenced '
            'the AI prediction. Red bars increase risk; blue bars reduce risk.')
        pdf.ln(2)
        try:
            pdf.image(shap_plot_path, x=12, w=180)
        except Exception:
            pass
        pdf.ln(3)

    # �-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-�
    # SECTION 5 — RECOMMENDATIONS
    # �-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-�
    pdf.section_title('Clinical Recommendations')
    recs = combined_result.get('recommendations', [])
    for i, rec in enumerate(recs, 1):
        pdf.recommendation_item(i, rec)

    pdf.ln(2)
    pdf.key_value('Urgency',            combined_result.get('urgency','N/A'))
    pdf.key_value('Recommended follow-up', combined_result.get('followup','N/A'))

    # Clinical notes
    if clinical_notes and clinical_notes.strip():
        pdf.ln(3)
        pdf.section_title('Physician Notes')
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(*TEXT)
        pdf.set_x(12)
        pdf.multi_cell(0, 6, clinical_notes.strip())

    # �-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-�
    # SECTION 6 — DISCLAIMER
    # �-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-��-�
    pdf.ln(4)
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(*BORDER)
    pdf.rect(10, pdf.get_y(), 190, 18, 'DF')
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(*MUTED)
    pdf.set_x(14)
    pdf.cell(0, 6, 'DISCLAIMER', ln=True)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_x(14)
    pdf.multi_cell(182, 5,
        'This report is generated by an AI system for educational and research purposes '
        'only. It does not constitute medical advice and must not be used for clinical '
        'diagnosis or treatment decisions. Always consult a qualified healthcare '
        'professional for medical evaluation.')

    # ── Save ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename  = f'reports/Hemo Check_{scan_type.replace(" ","_")}_{timestamp}.pdf'

    def clean_text(text):
        if not isinstance(text, str):
            return text

        replacements = {
            "—": "-",
            "–": "-",
            "•": "-",
            "…": "...",
            "’": "'",
            "‘": "'",
            "“": '"',
            "”": '"',
            "≤": "<=",
            "≥": ">=",
            "�-": "x",
            "°": " deg ",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    pdf.output(filename)
    print(f'Report saved: {filename}')
    return filename


# ── Backwards-compatible wrapper for clinical-only reports ─────────
def generate_report(patient_data: dict, prediction: int,
                    confidence: float, shap_plot_path: str = None) -> str:
    """Legacy wrapper — clinical-only report (no scan)."""
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
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(*MUTED)
    pdf.set_x(12)
    pdf.multi_cell(0, 5,
        'This report is for educational purposes only. '
        'Not a substitute for medical advice.')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename  = f'reports/Hemo Check_Clinical_{timestamp}.pdf'

    

    pdf.output(filename)
    return filename
