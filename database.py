"""
database.py — SQLite incident logger for FocusGuard sessions.
"""

import sqlite3
import os
import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'focusguard.db')

CREATE_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time  REAL    NOT NULL,
    end_time    REAL,
    duration    REAL,
    total_l1    INTEGER DEFAULT 0,
    total_l2    INTEGER DEFAULT 0,
    total_l3    INTEGER DEFAULT 0,
    avg_ear     REAL    DEFAULT 0,
    perclos_avg REAL    DEFAULT 0
);
"""

CREATE_INCIDENTS = """
CREATE TABLE IF NOT EXISTS incidents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL,
    timestamp   REAL    NOT NULL,
    alert_level INTEGER NOT NULL,
    reason      TEXT,
    ear_value   REAL,
    head_pose   TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
"""


@contextmanager
def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


class IncidentLogger:
    """Handles all DB operations for FocusGuard sessions and incidents."""

    def __init__(self):
        self._init_db()

    def _init_db(self):
        with get_conn() as conn:
            conn.execute(CREATE_SESSIONS)
            conn.execute(CREATE_INCIDENTS)
        logger.info(f"DB ready at {DB_PATH}")

    def start_session(self) -> int:
        with get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO sessions (start_time) VALUES (?)",
                (time.time(),)
            )
            return cur.lastrowid

    def log_incident(self, session_id: int, alert_level: int,
                     reason: str, ear: float, head_pose: str):
        with get_conn() as conn:
            conn.execute(
                """INSERT INTO incidents
                   (session_id, timestamp, alert_level, reason, ear_value, head_pose)
                   VALUES (?,?,?,?,?,?)""",
                (session_id, time.time(), alert_level, reason, ear, head_pose)
            )
            # Update session totals
            col = {1: "total_l1", 2: "total_l2", 3: "total_l3"}.get(alert_level)
            if col:
                conn.execute(
                    f"UPDATE sessions SET {col} = {col} + 1 WHERE id = ?",
                    (session_id,)
                )

    def end_session(self, session_id: int,
                    avg_ear: float = 0.0, perclos_avg: float = 0.0):
        now = time.time()
        with get_conn() as conn:
            row = conn.execute(
                "SELECT start_time FROM sessions WHERE id = ?",
                (session_id,)
            ).fetchone()
            duration = now - row["start_time"] if row else 0
            conn.execute(
                """UPDATE sessions
                   SET end_time=?, duration=?, avg_ear=?, perclos_avg=?
                   WHERE id=?""",
                (now, duration, avg_ear, perclos_avg, session_id)
            )

    def get_incidents(self, session_id: int = None,
                      limit: int = 50) -> list[dict]:
        with get_conn() as conn:
            if session_id:
                rows = conn.execute(
                    """SELECT * FROM incidents WHERE session_id=?
                       ORDER BY timestamp DESC LIMIT ?""",
                    (session_id, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM incidents ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def get_session_stats(self, session_id: int) -> dict:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,)
            ).fetchone()
            return dict(row) if row else {}

    def get_all_sessions(self) -> list[dict]:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY start_time DESC"
            ).fetchall()
            return [dict(r) for r in rows]
