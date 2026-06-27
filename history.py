"""
history.py
──────────
Patient prediction history using SQLite.
Stores every prediction with full metadata.
No external DB needed — runs as a local file.
"""

import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_PATH = 'data/Hemo Check_history.db'
os.makedirs('data', exist_ok=True)


def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT    NOT NULL,
            patient_id    TEXT    NOT NULL,
            scan_type     TEXT    NOT NULL,
            result_label  TEXT    NOT NULL,
            risk_level    TEXT    NOT NULL,
            confidence    REAL    NOT NULL,
            model_used    TEXT    NOT NULL,
            physician     TEXT,
            notes         TEXT,
            model_version TEXT    DEFAULT 'v3.0'
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(patient_id: str, scan_type: str, result_label: str,
                   risk_level: str, confidence: float, model_used: str,
                   physician: str = '', notes: str = ''):
    """Log a new prediction to the database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        INSERT INTO predictions
        (timestamp, patient_id, scan_type, result_label,
         risk_level, confidence, model_used, physician, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        patient_id, scan_type, result_label,
        risk_level, round(confidence, 2),
        model_used, physician, notes
    ))
    conn.commit()
    conn.close()


def get_all_history(limit: int = 50) -> pd.DataFrame:
    """Return recent predictions as a DataFrame."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(
        f"SELECT * FROM predictions ORDER BY id DESC LIMIT {limit}",
        conn
    )
    conn.close()
    return df


def get_patient_history(patient_id: str) -> pd.DataFrame:
    """Return all predictions for a specific patient."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(
        "SELECT * FROM predictions WHERE patient_id=? ORDER BY id DESC",
        conn, params=(patient_id,)
    )
    conn.close()
    return df


def get_stats() -> dict:
    """Return aggregate statistics for the analytics dashboard."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    total     = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    high_risk = c.execute(
        "SELECT COUNT(*) FROM predictions WHERE risk_level='HIGH'"
    ).fetchone()[0]
    avg_conf  = c.execute(
        "SELECT AVG(confidence) FROM predictions"
    ).fetchone()[0] or 0

    by_type = pd.read_sql_query(
        "SELECT scan_type, COUNT(*) as count FROM predictions GROUP BY scan_type",
        conn
    )
    conn.close()

    return {
        'total'       : total,
        'high_risk'   : high_risk,
        'avg_conf'    : round(avg_conf, 1),
        'by_type'     : by_type
    }


def clear_history():
    """Delete all records — for testing only."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()