"""
ear_calculator.py — Eye Aspect Ratio + PERCLOS drowsiness detection
Core algorithm for FocusGuard. Computes EAR from 6 eye landmarks.
"""

from collections import deque
import numpy as np
from scipy.spatial.distance import euclidean

# EAR thresholds
EAR_OPEN    = 0.25   # normal open eye
EAR_CLOSING = 0.20   # drooping — warning
EAR_CLOSED  = 0.15   # microsleep — danger

# PERCLOS thresholds (% of time eyes closed in last 60s)
PERCLOS_NORMAL  = 10.0
PERCLOS_WARNING = 15.0

# Landmark index sets for EAR (6 points per eye)
# Format: [P1(left), P2(top-left), P3(top-right), P4(right), P5(bot-right), P6(bot-left)]
LEFT_EYE_EAR  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR = [33,  160, 158, 133, 153, 144]


def _ear_from_points(pts: np.ndarray) -> float:
    """
    Eye Aspect Ratio = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
    pts: array of 6 (x, y) pixel coordinates in order P1..P6
    """
    p1, p2, p3, p4, p5, p6 = pts
    vertical_1 = euclidean(p2, p6)
    vertical_2 = euclidean(p3, p5)
    horizontal = euclidean(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


class EARCalculator:
    """
    Calculates Eye Aspect Ratio and PERCLOS from MediaPipe mesh points.
    Maintains a rolling history for PERCLOS computation.
    """

    def __init__(self, fps: int = 30, perclos_window_sec: int = 60):
        self.fps = fps
        self.history_size = fps * perclos_window_sec
        self.ear_history: deque = deque(maxlen=self.history_size)

    def calculate_ear(self, mesh_points: np.ndarray) -> tuple[float, float, float]:
        """
        Compute left EAR, right EAR, and average EAR.
        mesh_points: (N, 2) pixel coordinates from FaceDetector.
        Returns (left_ear, right_ear, avg_ear).
        """
        if mesh_points is None or len(mesh_points) < 468:
            return 0.0, 0.0, 0.0

        left_pts  = mesh_points[LEFT_EYE_EAR]
        right_pts = mesh_points[RIGHT_EYE_EAR]

        left_ear  = _ear_from_points(left_pts)
        right_ear = _ear_from_points(right_pts)
        avg_ear   = (left_ear + right_ear) / 2.0

        self.ear_history.append(avg_ear)
        return round(left_ear, 4), round(right_ear, 4), round(avg_ear, 4)

    def calculate_perclos(self) -> float:
        """
        PERCLOS = percentage of frames in history where EAR < EAR_CLOSING.
        Returns float 0–100.
        """
        if len(self.ear_history) < 10:
            return 0.0
        closed_count = sum(1 for e in self.ear_history if e < EAR_CLOSING)
        return round((closed_count / len(self.ear_history)) * 100, 2)

    def is_drowsy(self, ear: float, perclos: float) -> tuple[int, str]:
        """
        Classify drowsiness level from EAR and PERCLOS.
        Returns (level: int 0-3, reason: str).
        """
        if ear < EAR_CLOSED and perclos > PERCLOS_WARNING:
            return 3, f"Eyes closed (EAR={ear:.2f}) + PERCLOS {perclos:.1f}%"
        if ear < EAR_CLOSED:
            return 2, f"Eyes closed — EAR={ear:.2f}"
        if perclos > PERCLOS_WARNING:
            return 2, f"High PERCLOS={perclos:.1f}%"
        if ear < EAR_CLOSING or perclos > PERCLOS_NORMAL:
            return 1, f"Eyes drooping (EAR={ear:.2f}, PERCLOS={perclos:.1f}%)"
        return 0, "Normal"

    def get_drowsiness_score(self, perclos: float) -> int:
        """Return drowsiness score 0–100 for gauge display."""
        return min(100, int(perclos * 6.67))

    def reset(self):
        self.ear_history.clear()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import cv2
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from vision.face_detector import FaceDetector

    cap      = cv2.VideoCapture(0)
    detector = FaceDetector()
    calc     = EARCalculator()
    print("EAR test — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ann, landmarks, mesh_points, success = detector.process_frame(frame)

        if success:
            l, r, avg = calc.calculate_ear(mesh_points)
            perclos   = calc.calculate_perclos()
            level, reason = calc.is_drowsy(avg, perclos)

            colors = [(0,200,0),(0,200,255),(0,120,255),(0,0,255)]
            cv2.putText(ann, f"EAR L:{l:.3f} R:{r:.3f} Avg:{avg:.3f}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[level], 2)
            cv2.putText(ann, f"PERCLOS:{perclos:.1f}%  Level:{level}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[level], 2)
            cv2.putText(ann, reason, (10,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[level], 1)

        cv2.imshow("EAR Test", ann)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
