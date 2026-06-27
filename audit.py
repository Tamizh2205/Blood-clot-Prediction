"""
audit.py
────────
Full audit trail for every prediction made by the system.
Logs to CSV — readable, exportable, no extra dependencies.
"""

import pandas as pd
import os
from datetime import datetime

AUDIT_PATH = 'data/audit_log.csv'
os.makedirs('data', exist_ok=True)

COLUMNS = [
    'log_id', 'timestamp', 'patient_id', 'scan_type',
    'model_name', 'model_version', 'input_summary',
    'result_label', 'risk_level', 'confidence',
    'gradcam_generated', 'report_generated', 'physician'
]


def _init_log():
    if not os.path.exists(AUDIT_PATH):
        pd.DataFrame(columns=COLUMNS).to_csv(AUDIT_PATH, index=False)


def log_audit(patient_id: str, scan_type: str, model_name: str,
              result_label: str, risk_level: str, confidence: float,
              input_summary: str = '', gradcam: bool = False,
              report: bool = False, physician: str = '',
              model_version: str = 'v3.0') -> str:
    """
    Append one audit record and return the log_id.
    """
    _init_log()
    df     = pd.read_csv(AUDIT_PATH)
    log_id = f"LOG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(df)+1:04d}"

    new_row = pd.DataFrame([{
        'log_id'           : log_id,
        'timestamp'        : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'patient_id'       : patient_id,
        'scan_type'        : scan_type,
        'model_name'       : model_name,
        'model_version'    : model_version,
        'input_summary'    : input_summary,
        'result_label'     : result_label,
        'risk_level'       : risk_level,
        'confidence'       : round(confidence, 2),
        'gradcam_generated': gradcam,
        'report_generated' : report,
        'physician'        : physician
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(AUDIT_PATH, index=False)
    return log_id


def get_audit_log(limit: int = 100) -> pd.DataFrame:
    """Return the full audit log as a DataFrame."""
    _init_log()
    df = pd.read_csv(AUDIT_PATH)
    return df.tail(limit).iloc[::-1].reset_index(drop=True)


def export_audit_csv() -> str:
    """Return path to the audit CSV for download."""
    _init_log()
    return AUDIT_PATH