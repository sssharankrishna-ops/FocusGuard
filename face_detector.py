"""
face_detector.py — MediaPipe FaceMesh wrapper for FocusGuard
Detects facial landmarks and draws eye contours on frame.
"""

import cv2
import mediapipe as mp
import numpy as np


# MediaPipe landmark indices for eyes
LEFT_EYE_INDICES  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33,  7,   163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# Specific 6-point indices for EAR calculation (per eye)
LEFT_EAR_POINTS  = [362, 385, 387, 263, 373, 380]
RIGHT_EAR_POINTS = [33,  160, 158, 133, 153, 144]

# Lip landmarks for MAR (yawn)
LIP_POINTS = [13, 14, 78, 308]

# Head pose reference landmarks
HEAD_POSE_POINTS = [1, 152, 33, 263, 61, 291]


class FaceDetector:
    """
    Wraps MediaPipe FaceMesh for real-time facial landmark detection.
    Returns landmarks, pixel coordinates, and annotated frame.
    """

    def __init__(self, min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing   = mp.solutions.drawing_utils
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.success    = False
        self.landmarks  = None
        self.mesh_points = None

    def process_frame(self, frame: np.ndarray):
        """
        Process a BGR frame and detect face landmarks.

        Returns:
            annotated_frame: frame with eye contours drawn
            landmarks: normalized landmark list (MediaPipe format) or None
            mesh_points: (N, 2) pixel coordinates array or None
            success: bool
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        rgb.flags.writeable = True

        annotated = frame.copy()
        h, w = frame.shape[:2]

        if results.multi_face_landmarks:
            self.success  = True
            face_lm       = results.multi_face_landmarks[0]
            self.landmarks = face_lm.landmark

            # Convert to pixel coordinates
            self.mesh_points = np.array([
                [int(lm.x * w), int(lm.y * h)]
                for lm in face_lm.landmark
            ], dtype=np.int32)

            # Draw eye contours
            self._draw_eye_contour(annotated, LEFT_EYE_INDICES,  (0, 255, 120))
            self._draw_eye_contour(annotated, RIGHT_EYE_INDICES, (0, 255, 120))

        else:
            self.success     = False
            self.landmarks   = None
            self.mesh_points = None

        return annotated, self.landmarks, self.mesh_points, self.success

    def _draw_eye_contour(self, frame, indices, color):
        if self.mesh_points is None:
            return
        pts = self.mesh_points[indices]
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1)

    def get_pixel_coords(self, indices: list) -> np.ndarray:
        """Return pixel coords for a list of landmark indices."""
        if self.mesh_points is None:
            return np.array([])
        return self.mesh_points[indices]

    def release(self):
        self.face_mesh.close()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    print("Face detector test — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, landmarks, mesh_points, success = detector.process_frame(frame)

        status = "Face detected" if success else "No face"
        cv2.putText(annotated, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if success else (0, 0, 255), 2)

        cv2.imshow("FocusGuard — Face Detector Test", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.release()
    cap.release()
    cv2.destroyAllWindows()
