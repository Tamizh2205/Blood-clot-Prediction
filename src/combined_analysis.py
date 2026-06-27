"""
combined_analysis.py
────────────────────
Fuses scan model output + clinical ensemble output + symptom score
into a single unified risk assessment with confidence and
actionable recommendations.
"""

from __future__ import annotations
import datetime


# ── Weights for fusion ─────────────────────────────────────────────
SCAN_WEIGHT     = 0.50
CLINICAL_WEIGHT = 0.35
SYMPTOM_WEIGHT  = 0.15


def fuse_results(
    scan_result      : dict,
    clinical_result  : dict,
    symptom_result   : dict,
    scan_type        : str,
) -> dict:
    """
    Combine scan, clinical, and symptom scores into one assessment.

    Parameters
    ----------
    scan_result      : dict from ct_detector / mri_detector / ultrasound_detector
    clinical_result  : dict from ensemble_predict()
    symptom_result   : dict from symptom_risk_score()
    scan_type        : "CT Scan" | "MRI Brain" | "Ultrasound DVT"

    Returns
    -------
    dict with keys:
      overall_probability, overall_confidence, overall_risk,
      overall_label, color, recommendations, summary_lines,
      scan_contribution, clinical_contribution, symptom_contribution,
      generated_at
    """

    scan_prob     = scan_result.get('raw_prob', 0.5)
    clinical_prob = clinical_result.get('probability', 0.5)
    symptom_prob  = symptom_result.get('score', 0) / 100.0

    overall_prob  = (
        SCAN_WEIGHT     * scan_prob +
        CLINICAL_WEIGHT * clinical_prob +
        SYMPTOM_WEIGHT  * symptom_prob
    )
    overall_conf  = round(overall_prob * 100, 1)

    # ── Risk classification ────────────────────────────────────────
    if overall_conf >= 70:
        overall_risk  = "HIGH"
        overall_label = "High Blood Clot Risk Detected"
        color         = "#be123c"
        kind          = "danger"
    elif overall_conf >= 50:
        overall_risk  = "MEDIUM"
        overall_label = "Moderate Blood Clot Risk — Monitor"
        color         = "#d97706"
        kind          = "warn"
    elif overall_conf >= 30:
        overall_risk  = "LOW"
        overall_label = "Low Blood Clot Risk"
        color         = "#15803d"
        kind          = "safe"
    else:
        overall_risk  = "VERY LOW"
        overall_label = "Very Low Blood Clot Risk"
        color         = "#0891b2"
        kind          = "safe"

    if overall_conf < 55 and overall_conf > 40:
        overall_risk  = "INCONCLUSIVE"
        overall_label = "Inconclusive — Further Review Required"
        color         = "#64748b"
        kind          = "neutral"

    # ── Scan-specific language ─────────────────────────────────────
    scan_finding = {
        "CT Scan"      : "pulmonary embolism indicators",
        "MRI Brain"    : "brain clot or stroke indicators",
        "Ultrasound DVT": "deep vein thrombosis markers",
    }.get(scan_type, "imaging abnormalities")

    # ── Recommendations ────────────────────────────────────────────
    if overall_risk == "HIGH":
        recs = [
            f"Urgent clinical evaluation for {scan_finding}.",
            "Immediate physician referral — do not delay.",
            "Consider anticoagulation therapy consultation.",
            "Repeat imaging within 24 hours if clinically indicated.",
            "Review all current medications with treating physician.",
            "Patient should not travel or remain immobile.",
        ]
        urgency  = "URGENT — within 24 hours"
        followup = "24–48 hours with specialist"

    elif overall_risk == "MEDIUM":
        recs = [
            f"Elevated markers for {scan_finding} detected.",
            "Schedule follow-up imaging within 1 week.",
            "Begin lifestyle modifications: movement, hydration.",
            "Review risk factors with primary care physician.",
            "Consider D-Dimer blood test for further confirmation.",
        ]
        urgency  = "PROMPT — within 1 week"
        followup = "1 week with primary care physician"

    elif overall_risk == "INCONCLUSIVE":
        recs = [
            "Combined confidence below diagnostic threshold.",
            "Repeat scan under better imaging conditions.",
            "Obtain additional clinical history and lab work.",
            "Specialist consultation recommended.",
        ]
        urgency  = "AS SOON AS POSSIBLE"
        followup = "Within 48 hours for repeat assessment"

    else:
        recs = [
            f"No significant {scan_finding} detected at this time.",
            "Maintain healthy lifestyle and regular exercise.",
            "Monitor blood pressure and cholesterol every 6 months.",
            "Routine annual clinical review recommended.",
            "Return if symptoms develop or worsen.",
        ]
        urgency  = "ROUTINE — annual review"
        followup = "12 months (routine checkup)"

    # ── Contribution breakdown ─────────────────────────────────────
    scan_contrib     = round(scan_prob     * SCAN_WEIGHT     * 100, 1)
    clinical_contrib = round(clinical_prob * CLINICAL_WEIGHT * 100, 1)
    symptom_contrib  = round(symptom_prob  * SYMPTOM_WEIGHT  * 100, 1)

    # ── Summary lines for report ───────────────────────────────────
    summary_lines = [
        ("Overall risk classification", overall_risk,
         "danger" if overall_risk == "HIGH" else
         "warn"   if overall_risk == "MEDIUM" else "safe"),
        ("Overall confidence",          f"{overall_conf}%",       ""),
        ("Scan model contribution",     f"{scan_contrib}%",       ""),
        ("Clinical model contribution", f"{clinical_contrib}%",   ""),
        ("Symptom score contribution",  f"{symptom_contrib}%",    ""),
        ("Scan finding",
         scan_result.get('label', 'N/A'),
         "danger" if scan_result.get('risk_level') == 'HIGH' else ""),
        ("Scan confidence",
         f"{scan_result.get('confidence', 0)}%",                  ""),
        ("Clinical risk level",
         clinical_result.get('risk_level', 'N/A'),
         "danger" if clinical_result.get('risk_level') == 'HIGH' else ""),
        ("Clinical confidence",
         f"{clinical_result.get('confidence', 0)}%",              ""),
        ("Symptom severity",
         f"{symptom_result.get('label', 'NONE')} "
         f"({symptom_result.get('score', 0)}/100)",               ""),
        ("Urgency",                     urgency,
         "danger" if overall_risk == "HIGH" else ""),
        ("Recommended follow-up",       followup,                 ""),
    ]

    return {
        'overall_probability'  : overall_prob,
        'overall_confidence'   : overall_conf,
        'overall_risk'         : overall_risk,
        'overall_label'        : overall_label,
        'color'                : color,
        'kind'                 : kind,
        'recommendations'      : recs,
        'summary_lines'        : summary_lines,
        'urgency'              : urgency,
        'followup'             : followup,
        'scan_contribution'    : scan_contrib,
        'clinical_contribution': clinical_contrib,
        'symptom_contribution' : symptom_contrib,
        'generated_at'         : datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


def get_risk_color(rl: str) -> str:
    return {
        'HIGH'       : '#be123c',
        'MEDIUM'     : '#d97706',
        'LOW'        : '#15803d',
        'VERY LOW'   : '#0891b2',
        'INCONCLUSIVE': '#64748b',
    }.get(rl, '#64748b')


def get_risk_kind(rl: str) -> str:
    return {
        'HIGH'       : 'danger',
        'MEDIUM'     : 'warn',
        'LOW'        : 'safe',
        'VERY LOW'   : 'safe',
        'INCONCLUSIVE': 'neutral',
    }.get(rl, 'neutral')