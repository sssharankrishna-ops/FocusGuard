"""
yawn_detector.py — Yawn detection via Mouth Aspect Ratio (MAR)
Secondary drowsiness signal for FocusGuard.
"""

import time
from collections import deque
import numpy as np
from scipy.spatial.distance import euclidean

# MediaPipe lip landmark indices
LIP_TOP    = 13
LIP_BOTTOM = 14
LIP_LEFT   = 78
LIP_RIGHT  = 308

# Thresholds
MAR_THRESHOLD    = 0.6    # mouth open ratio to classify as yawn
YAWN_DURATION    = 1.5    # seconds mouth must be open to count as yawn
YAWN_WINDOW      = 600    # seconds (10 minutes) to count yawns in
FATIGUE_YAWNS_1  = 2      # yawns → level 1 fatigue signal
FATIGUE_YAWNS_2  = 4      # yawns → level 2 fatigue signal


class YawnDetector:
    """
    Detects yawns using Mouth Aspect Ratio from MediaPipe landmarks.
    Tracks yawn count over the last 10 minutes.
    """

    def __init__(self):
        self._yawn_start: float | None = None
        self._is_yawning = False
        self._yawn_timestamps: deque = deque()  # timestamps of completed yawns

    def _mar(self, mesh_points: np.ndarray) -> float:
        """Compute Mouth Aspect Ratio."""
        if mesh_points is None or len(mesh_points) < 468:
            return 0.0
        top    = mesh_points[LIP_TOP]
        bottom = mesh_points[LIP_BOTTOM]
        left   = mesh_points[LIP_LEFT]
        right  = mesh_points[LIP_RIGHT]

        vertical   = euclidean(top, bottom)
        horizontal = euclidean(left, right)
        if horizontal == 0:
            return 0.0
        return round(vertical / horizontal, 4)

    def detect(self, mesh_points: np.ndarray) -> tuple[bool, float]:
        """
        Detect yawn state from mesh_points.
        Returns (is_yawning: bool, mar: float).
        A yawn is counted when MAR stays above threshold for YAWN_DURATION seconds.
        """
        mar = self._mar(mesh_points)
        now = time.time()

        if mar > MAR_THRESHOLD:
            if self._yawn_start is None:
                self._yawn_start = now
            elif now - self._yawn_start >= YAWN_DURATION and not self._is_yawning:
                self._is_yawning = True
                self._yawn_timestamps.append(now)
        else:
            self._yawn_start  = None
            self._is_yawning  = False

        return self._is_yawning, mar

    def get_recent_yawn_count(self) -> int:
        """Return number of yawns in the last 10 minutes."""
        now    = time.time()
        cutoff = now - YAWN_WINDOW
        # Remove old yawns
        while self._yawn_timestamps and self._yawn_timestamps[0] < cutoff:
            self._yawn_timestamps.popleft()
        return len(self._yawn_timestamps)

    def get_fatigue_from_yawns(self) -> int:
        """
        Return fatigue level 0–2 based on yawn frequency.
        0 = normal, 1 = mild fatigue, 2 = significant fatigue.
        """
        count = self.get_recent_yawn_count()
        if count >= FATIGUE_YAWNS_2:
            return 2
        if count >= FATIGUE_YAWNS_1:
            return 1
        return 0

    def reset(self):
        self._yawn_start = None
        self._is_yawning = False
        self._yawn_timestamps.clear()
