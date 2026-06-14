"""
main.py — FocusGuard Vision Engine Entry Point
Orchestrates all detectors and streams state to WebSocket server.
"""

import cv2
import asyncio
import time
import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vision.face_detector  import FaceDetector
from vision.ear_calculator import EARCalculator
from vision.head_pose      import HeadPoseEstimator
from vision.yawn_detector  import YawnDetector
from vision.phone_detector import PhoneDetector
from vision.alert_engine   import AlertEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s — %(message)s'
)
logger = logging.getLogger("focusguard.main")

# ── State broadcast ───────────────────────────────────────────────────────────
# This queue is imported by server/app.py to broadcast via WebSocket
state_queue: asyncio.Queue = asyncio.Queue(maxsize=5)


def put_state(state: dict):
    """Non-blocking put — drop oldest if full."""
    try:
        state_queue.put_nowait(state)
    except asyncio.QueueFull:
        try:
            state_queue.get_nowait()
            state_queue.put_nowait(state)
        except Exception:
            pass


# ── HUD overlay ───────────────────────────────────────────────────────────────
def draw_hud(frame, state: dict, alert_color):
    h, w = frame.shape[:2]

    # Dark top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # EAR
    cv2.putText(frame,
                f"EAR  L:{state['ear_left']:.3f}  R:{state['ear_right']:.3f}  Avg:{state['ear_avg']:.3f}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220,220,220), 1)

    # PERCLOS bar
    perclos = state['perclos']
    bar_w   = int((w - 20) * perclos / 100)
    cv2.rectangle(frame, (10, 38), (w - 10, 55), (60,60,60), -1)
    bar_color = (0,200,0) if perclos < 10 else (0,180,255) if perclos < 15 else (0,0,255)
    cv2.rectangle(frame, (10, 38), (10 + bar_w, 55), bar_color, -1)
    cv2.putText(frame, f"PERCLOS {perclos:.1f}%",
                (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)

    # Head pose
    cv2.putText(frame,
                f"Head  P:{state['head_pitch']:.1f}  Y:{state['head_yaw']:.1f}  R:{state['head_roll']:.1f}  [{state['head_state']}]",
                (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

    # Alert badge
    level      = state['alert_level']
    badge_text = ["NORMAL","WARNING","DANGER","CRITICAL"][level]
    badge_col  = [(0,180,0),(0,180,255),(0,80,255),(0,0,220)][level]
    tx, ty     = w - 170, 18
    cv2.rectangle(frame, (tx - 8, 4), (w - 4, 36), badge_col, -1)
    cv2.putText(frame, badge_text, (tx, ty + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

    # FPS
    cv2.putText(frame, f"FPS {state['fps']:.0f}",
                (w - 80, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160,160,160), 1)

    # Flash overlay for level 2/3
    if level >= 2:
        flash = frame.copy()
        cv2.rectangle(flash, (0, 0), (w, h), (0, 0, 200), -1)
        alpha = 0.15 if level == 2 else 0.25
        cv2.addWeighted(flash, alpha, frame, 1 - alpha, 0, frame)

    return frame


# ── Main capture loop ─────────────────────────────────────────────────────────
def run(camera_idx: int = 0, sensitivity: str = "medium",
        no_display: bool = False):

    # Sensitivity → EAR thresholds adjustment
    ear_offset = {"low": 0.02, "medium": 0.0, "high": -0.02}[sensitivity]

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {camera_idx}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    face_det   = FaceDetector()
    ear_calc   = EARCalculator()
    pose_est   = HeadPoseEstimator()
    yawn_det   = YawnDetector()
    phone_det  = PhoneDetector()
    alerter    = AlertEngine()

    # Import DB logger lazily (server may not be running)
    try:
        from server.database import IncidentLogger
        db = IncidentLogger()
        session_id = db.start_session()
        logger.info(f"Session {session_id} started.")
    except Exception:
        db, session_id = None, None
        logger.warning("DB not available — incidents not persisted.")

    prev_time = time.time()
    frame_n   = 0

    logger.info("FocusGuard vision engine running — press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Frame capture failed.")
            break

        frame_n += 1
        now      = time.time()
        fps      = 1.0 / max(now - prev_time, 0.001)
        prev_time = now

        # ── Face detection ────────────────────────────────────────────────
        ann, landmarks, mesh_pts, face_ok = face_det.process_frame(frame)

        if face_ok:
            # EAR
            l_ear, r_ear, avg_ear = ear_calc.calculate_ear(mesh_pts)
            perclos               = ear_calc.calculate_perclos()
            ear_level, ear_reason = ear_calc.is_drowsy(avg_ear + ear_offset,
                                                        perclos)

            # Head pose
            pitch, yaw, roll        = pose_est.estimate(mesh_pts, frame.shape)
            head_state, distracted  = pose_est.classify_pose(pitch, yaw, roll)
            ann = pose_est.draw_pose_arrow(ann, mesh_pts, distracted)

            # Yawn
            is_yawning, mar = yawn_det.detect(mesh_pts)
            yawn_count      = yawn_det.get_recent_yawn_count()
            yawn_fatigue    = yawn_det.get_fatigue_from_yawns()

            # Phone
            phone_dist, phone_conf, phone_bbox = phone_det.detect(frame)
            ann = phone_det.draw_detection(ann, phone_bbox, phone_conf)

            # ── Compute final alert level ─────────────────────────────────
            alert_level = ear_level
            alert_reason = ear_reason

            if distracted and yaw > 25:
                lvl = 2 if yaw > 35 else 1
                if lvl > alert_level:
                    alert_level  = lvl
                    alert_reason = f"Head distraction (yaw={yaw:.1f}°)"
            if phone_dist:
                if 2 > alert_level:
                    alert_level  = 2
                    alert_reason = "Phone detected"
            if yawn_fatigue > 0 and yawn_fatigue > alert_level:
                alert_level  = yawn_fatigue
                alert_reason = f"Yawn fatigue ({yawn_count} yawns)"

            # Trigger alert + log
            triggered = alerter.trigger_alert(alert_level, alert_reason)
            if triggered and db and session_id:
                try:
                    db.log_incident(session_id, alert_level, alert_reason,
                                    avg_ear, head_state)
                except Exception:
                    pass

        else:
            l_ear = r_ear = avg_ear = perclos = 0.0
            pitch = yaw = roll = 0.0
            head_state = "no_face"
            distracted = is_yawning = phone_dist = False
            mar = phone_conf = 0.0
            yawn_count = 0
            alert_level = 0
            alert_reason = "No face detected"

        # ── Build state dict ──────────────────────────────────────────────
        state = {
            "ear_left":    l_ear,
            "ear_right":   r_ear,
            "ear_avg":     avg_ear,
            "perclos":     perclos,
            "head_pitch":  pitch,
            "head_yaw":    yaw,
            "head_roll":   roll,
            "head_state":  head_state,
            "is_distracted": distracted,
            "is_yawning":  is_yawning,
            "yawn_count":  yawn_count,
            "phone_detected": phone_dist,
            "alert_level": alert_level,
            "alert_reason": alert_reason,
            "face_detected": face_ok,
            "fps":         round(fps, 1),
            "timestamp":   now,
        }
        put_state(state)

        # ── Display ───────────────────────────────────────────────────────
        if not no_display:
            ann = draw_hud(ann, state, alerter.get_alert_color_bgr())
            cv2.imshow("FocusGuard", ann)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # ── Cleanup ───────────────────────────────────────────────────────────
    logger.info("Shutting down FocusGuard.")
    alerter.shutdown()
    face_det.release()
    cap.release()
    if not no_display:
        cv2.destroyAllWindows()
    if db and session_id:
        db.end_session(session_id)


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FocusGuard Vision Engine")
    parser.add_argument("--camera",      type=int,   default=0,
                        help="Camera index (default 0)")
    parser.add_argument("--sensitivity", type=str,   default="medium",
                        choices=["low","medium","high"],
                        help="Alert sensitivity")
    parser.add_argument("--no-display",  action="store_true",
                        help="Run headless (no cv2.imshow window)")
    args = parser.parse_args()
    run(args.camera, args.sensitivity, args.no_display)
