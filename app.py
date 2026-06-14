"""
app.py — FastAPI + WebSocket server for FocusGuard.
Receives state from vision engine and broadcasts to React dashboard.
"""

import asyncio
import json
import time
import logging
import os
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)

app = FastAPI(title="FocusGuard API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state ──────────────────────────────────────────────────────────────
latest_state: dict = {
    "ear_left": 0.0, "ear_right": 0.0, "ear_avg": 0.0,
    "perclos": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0,
    "head_state": "unknown", "is_distracted": False, "is_yawning": False,
    "yawn_count": 0, "phone_detected": False, "alert_level": 0,
    "alert_reason": "Starting…", "face_detected": False,
    "fps": 0.0, "timestamp": time.time(),
    "drowsiness_score": 0,
}

connected_clients: list[WebSocket] = []
current_session_id: int | None     = None
session_start_time: float          = time.time()
session_ear_samples: list          = []

# ── DB ────────────────────────────────────────────────────────────────────────
try:
    from server.database import IncidentLogger
    db             = IncidentLogger()
    current_session_id = db.start_session()
    logger.info(f"Session {current_session_id} started in DB.")
except Exception as e:
    db = None
    logger.warning(f"DB unavailable: {e}")


# ── State ingestion endpoint (called by vision/main.py via HTTP) ──────────────
@app.post("/api/ingest")
async def ingest_state(payload: dict):
    """Vision engine POSTs state here; server broadcasts to WS clients."""
    global latest_state, session_ear_samples

    # Compute drowsiness score from PERCLOS
    perclos = payload.get("perclos", 0)
    payload["drowsiness_score"] = min(100, int(perclos * 6.67))
    latest_state = payload

    # Track EAR for session average
    if payload.get("face_detected") and payload.get("ear_avg", 0) > 0:
        session_ear_samples.append(payload["ear_avg"])

    # Log incidents
    level = payload.get("alert_level", 0)
    if level > 0 and db and current_session_id:
        try:
            db.log_incident(
                current_session_id, level,
                payload.get("alert_reason", ""),
                payload.get("ear_avg", 0),
                payload.get("head_state", "")
            )
        except Exception:
            pass

    # Broadcast to all WS clients
    await _broadcast(payload)
    return {"ok": True}


async def _broadcast(data: dict):
    dead = []
    for ws in connected_clients:
        try:
            await ws.send_text(json.dumps(data))
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_clients.remove(ws)


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws/state")
async def websocket_state(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    logger.info(f"WS client connected. Total: {len(connected_clients)}")

    # Send current state immediately on connect
    await websocket.send_text(json.dumps(latest_state))

    try:
        while True:
            # Keep connection alive; server pushes via broadcast
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        logger.info(f"WS client disconnected. Remaining: {len(connected_clients)}")


# ── REST endpoints ────────────────────────────────────────────────────────────
@app.get("/api/session/current")
def get_current_session():
    duration   = time.time() - session_start_time
    avg_ear    = (sum(session_ear_samples) / len(session_ear_samples)
                  if session_ear_samples else 0.0)
    stats = {
        "session_id":    current_session_id,
        "start_time":    session_start_time,
        "duration_sec":  round(duration, 1),
        "avg_ear":       round(avg_ear, 4),
        "total_samples": len(session_ear_samples),
    }
    if db and current_session_id:
        db_stats = db.get_session_stats(current_session_id)
        stats.update(db_stats)
    return stats


@app.get("/api/incidents")
def get_incidents(limit: int = 50):
    if db:
        return db.get_incidents(limit=limit)
    return []


@app.get("/api/sessions")
def get_sessions():
    if db:
        return db.get_all_sessions()
    return []


@app.post("/api/session/end")
def end_session():
    global current_session_id, session_start_time, session_ear_samples
    if db and current_session_id:
        avg_ear = (sum(session_ear_samples) / len(session_ear_samples)
                   if session_ear_samples else 0.0)
        db.end_session(current_session_id, avg_ear=avg_ear)

        # Trigger report generation
        try:
            from reports.report_generator import generate_report
            report_path = generate_report(current_session_id)
            logger.info(f"Report saved: {report_path}")
            result = {"ok": True, "report": report_path,
                      "session_id": current_session_id}
        except Exception as e:
            result = {"ok": True, "report": None, "error": str(e)}

        session_ear_samples   = []
        session_start_time    = time.time()
        current_session_id    = db.start_session()
        return result

    return {"ok": False, "error": "No active session"}


@app.get("/api/state")
def get_state():
    return latest_state


@app.get("/")
def root():
    return {"service": "FocusGuard API", "status": "running",
            "dashboard": "http://localhost:3000"}


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)
