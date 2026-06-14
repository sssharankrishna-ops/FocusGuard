"""
phone_detector.py — Phone detection using MobileNet SSD (OpenCV DNN)
Detects phone use as a distraction signal. Graceful fallback if model missing.
"""

import cv2
import numpy as np
import os
import time
import logging

logger = logging.getLogger(__name__)

MODEL_DIR     = os.path.join(os.path.dirname(__file__), '..', 'models', 'mobilenet_ssd')
PROTOTXT      = os.path.join(MODEL_DIR, 'deploy.prototxt')
CAFFEMODEL    = os.path.join(MODEL_DIR, 'mobilenet_iter_73000.caffemodel')

CONFIDENCE_THRESHOLD = 0.65
PHONE_CLASS_ID       = 77   # COCO class index for 'cell phone'
PHONE_DURATION       = 1.5  # seconds of continuous detection to trigger alert
SKIP_FRAMES          = 5    # run detection every N frames to save CPU

COCO_CLASSES = {
    0:'background',1:'aeroplane',2:'bicycle',3:'bird',4:'boat',5:'bottle',
    6:'bus',7:'car',8:'cat',9:'chair',10:'cow',11:'diningtable',12:'dog',
    13:'horse',14:'motorbike',15:'person',16:'pottedplant',17:'sheep',
    18:'sofa',19:'train',20:'tvmonitor',
    # SSD MobileNet uses different index for phone — handle both datasets
}

class PhoneDetector:
    """
    Detects mobile phone usage using MobileNet SSD.
    Falls back gracefully when model files are not present.
    """

    def __init__(self):
        self._net          = None
        self._available    = False
        self._frame_count  = 0
        self._phone_since: float | None = None
        self._last_result  = False
        self._last_conf    = 0.0
        self._last_bbox    = (0, 0, 0, 0)

        self._load_model()

    def _load_model(self):
        if os.path.exists(PROTOTXT) and os.path.exists(CAFFEMODEL):
            try:
                self._net       = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
                self._available = True
                logger.info("MobileNet SSD phone detector loaded.")
            except Exception as e:
                logger.warning(f"Failed to load phone detector model: {e}")
                self._available = False
        else:
            logger.warning(
                "Phone detector model files not found in models/mobilenet_ssd/. "
                "Download from: https://github.com/chuanqi305/MobileNet-SSD "
                "Phone detection disabled — all other alerts still active."
            )
            self._available = False

    def detect(self, frame: np.ndarray) -> tuple[bool, float, tuple]:
        """
        Detect phone in frame.
        Returns (is_distracted: bool, confidence: float, bbox: (x,y,w,h)).
        is_distracted is True only if phone visible for > PHONE_DURATION seconds.
        """
        self._frame_count += 1

        # Run inference every SKIP_FRAMES frames
        if self._frame_count % SKIP_FRAMES != 0:
            return self._is_distracted(), self._last_conf, self._last_bbox

        if not self._available:
            return False, 0.0, (0, 0, 0, 0)

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5
        )
        self._net.setInput(blob)
        detections = self._net.forward()

        phone_detected = False
        best_conf      = 0.0
        best_bbox      = (0, 0, 0, 0)

        for i in range(detections.shape[2]):
            conf  = float(detections[0, 0, i, 2])
            cls   = int(detections[0, 0, i, 1])

            # SSD MobileNet VOC: no phone class. COCO version has class 77.
            # We check confidence on any detected object near face region as proxy.
            if conf > CONFIDENCE_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                bw, bh = x2 - x1, y2 - y1

                # Heuristic: small rectangle in upper frame = phone near face
                if (conf > best_conf and bh > 30 and bw > 15
                        and y1 < h * 0.7):
                    best_conf      = conf
                    best_bbox      = (x1, y1, bw, bh)
                    phone_detected = True

        self._last_conf = best_conf
        self._last_bbox = best_bbox
        now = time.time()

        if phone_detected:
            if self._phone_since is None:
                self._phone_since = now
        else:
            self._phone_since = None

        self._last_result = phone_detected
        return self._is_distracted(), best_conf, best_bbox

    def _is_distracted(self) -> bool:
        if self._phone_since is None:
            return False
        return (time.time() - self._phone_since) >= PHONE_DURATION

    def draw_detection(self, frame: np.ndarray, bbox: tuple,
                       conf: float) -> np.ndarray:
        if conf > 0 and bbox != (0, 0, 0, 0):
            x, y, w, h = bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, f"PHONE {conf:.0%}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame
