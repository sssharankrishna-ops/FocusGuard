"""
head_pose.py — 3D Head Pose Estimation using solvePnP
Detects forward slump, left/right distraction for FocusGuard.
"""

import cv2
import numpy as np

# 3D model reference points (generic face model, mm)
MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),     # Nose tip          — landmark 1
    (0.0,   -130.0, -30.0),   # Chin              — landmark 152
    (-165.0, 170.0, -135.0),  # Left eye corner   — landmark 33
    (165.0,  170.0, -135.0),  # Right eye corner  — landmark 263
    (-150.0, -150.0, -125.0), # Left mouth corner — landmark 61
    (150.0,  -150.0, -125.0), # Right mouth corner— landmark 291
], dtype=np.float64)

# Corresponding MediaPipe landmark indices
POSE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# Distraction thresholds (degrees)
YAW_THRESHOLD   = 25.0   # left/right look
PITCH_THRESHOLD = 15.0   # head down
ROLL_THRESHOLD  = 20.0   # tilt


class HeadPoseEstimator:
    """
    Estimates pitch, yaw, roll of driver's head using solvePnP.
    Classifies pose state and draws direction arrow on frame.
    """

    def __init__(self):
        self._camera_matrix = None
        self._dist_coeffs   = np.zeros((4, 1))

    def _get_camera_matrix(self, frame_shape):
        h, w = frame_shape[:2]
        focal = w
        cx, cy = w / 2, h / 2
        return np.array([
            [focal, 0,     cx],
            [0,     focal, cy],
            [0,     0,     1 ]
        ], dtype=np.float64)

    def estimate(self, mesh_points: np.ndarray,
                 frame_shape: tuple) -> tuple[float, float, float]:
        """
        Estimate head pose from mesh pixel coordinates.
        Returns (pitch, yaw, roll) in degrees.
        Positive pitch = head down; positive yaw = looking right.
        """
        if mesh_points is None or len(mesh_points) < 468:
            return 0.0, 0.0, 0.0

        cam = self._get_camera_matrix(frame_shape)
        img_pts = np.array([
            mesh_points[i] for i in POSE_LANDMARK_IDS
        ], dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(
            MODEL_POINTS, img_pts, cam, self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0, 0.0

        rot_mat, _ = cv2.Rodrigues(rot_vec)
        angles, *_ = cv2.RQDecomp3x3(rot_mat)
        pitch = angles[0] * 360
        yaw   = angles[1] * 360
        roll  = angles[2] * 360

        # Store for arrow drawing
        self._rot_vec   = rot_vec
        self._trans_vec = trans_vec
        self._cam       = cam

        return round(pitch, 2), round(yaw, 2), round(roll, 2)

    def classify_pose(self, pitch: float, yaw: float,
                      roll: float) -> tuple[str, bool]:
        """
        Classify head pose state and whether driver is distracted.
        Returns (state_str, is_distracted).
        """
        if abs(yaw) > YAW_THRESHOLD:
            direction  = "right" if yaw > 0 else "left"
            return f"looking_{direction}", True
        if pitch > PITCH_THRESHOLD:
            return "head_down", True
        if abs(roll) > ROLL_THRESHOLD:
            return "head_tilted", True
        return "forward", False

    def draw_pose_arrow(self, frame: np.ndarray,
                        mesh_points: np.ndarray,
                        is_distracted: bool) -> np.ndarray:
        """Draw a 3D nose direction arrow on the frame."""
        if not hasattr(self, '_rot_vec') or mesh_points is None:
            return frame

        color = (0, 0, 255) if is_distracted else (0, 255, 0)
        nose_tip = tuple(mesh_points[1])

        # Project a point in front of the nose
        forward_pt = np.array([[0.0, 0.0, -100.0]])
        proj, _ = cv2.projectPoints(
            forward_pt, self._rot_vec, self._trans_vec,
            self._cam, self._dist_coeffs
        )
        proj_pt = tuple(proj[0][0].astype(int))

        cv2.arrowedLine(frame, nose_tip, proj_pt, color, 3, tipLength=0.3)
        return frame


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from vision.face_detector import FaceDetector

    cap      = cv2.VideoCapture(0)
    detector = FaceDetector()
    poser    = HeadPoseEstimator()
    print("Head pose test — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ann, _, mesh_points, success = detector.process_frame(frame)

        if success:
            pitch, yaw, roll = poser.estimate(mesh_points, frame.shape)
            state, distracted = poser.classify_pose(pitch, yaw, roll)
            ann = poser.draw_pose_arrow(ann, mesh_points, distracted)

            color = (0, 0, 255) if distracted else (0, 255, 0)
            cv2.putText(ann, f"Pitch:{pitch:.1f}  Yaw:{yaw:.1f}  Roll:{roll:.1f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(ann, f"State: {state}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Head Pose Test", ann)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
