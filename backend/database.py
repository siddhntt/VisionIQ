"""SQLite storage for analysis results."""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE_PATH


def get_connection(db_path=DATABASE_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=DATABASE_PATH):
    conn = get_connection(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            filename      TEXT NOT NULL,
            quality_score REAL NOT NULL,
            quality_label TEXT NOT NULL,
            confidence    REAL NOT NULL,
            issues_json   TEXT NOT NULL DEFAULT '[]',
            image_stats   TEXT NOT NULL DEFAULT '{}',
            model_signals TEXT NOT NULL DEFAULT '{}',
            heatmap       TEXT DEFAULT '',
            created_at    TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_analysis(filename, result, db_path=DATABASE_PATH):
    conn = get_connection(db_path)
    cur = conn.execute(
        """INSERT INTO analyses
           (filename, quality_score, quality_label, confidence,
            issues_json, image_stats, model_signals, heatmap, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (filename, result["quality_score"], result["quality_label"],
         result["confidence"], json.dumps(result.get("issues", [])),
         json.dumps(result.get("image_stats", {})),
         json.dumps(result.get("model_signals", {})),
         result.get("heatmap", ""),
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_analysis(analysis_id, db_path=DATABASE_PATH):
    conn = get_connection(db_path)
    row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def get_all_analyses(limit=50, offset=0, db_path=DATABASE_PATH):
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM analyses ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)
    ).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def get_analysis_count(db_path=DATABASE_PATH):
    conn = get_connection(db_path)
    count = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    conn.close()
    return count


def _row_to_dict(row):
    d = dict(row)
    d["issues"] = json.loads(d.pop("issues_json", "[]"))
    d["image_stats"] = json.loads(d.pop("image_stats", "{}"))
    d["model_signals"] = json.loads(d.pop("model_signals", "{}"))
    return d
