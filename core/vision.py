"""
core/vision.py — Hand detection and landmark normalization.

Uses the MediaPipe Tasks API (mediapipe >= 0.10.x) which replaced
the legacy mp.solutions.hands interface.

Provides:
  - HandDetector: open/close lifecycle + per-frame detection
  - normalize_landmarks(): pure function — wrist-relative, scale-invariant
"""

import logging
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import mediapipe as mp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model file management — download once, cache locally
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODEL_DIR = _PROJECT_ROOT / "checkpoints"
_HAND_MODEL_PATH = _MODEL_DIR / "hand_landmarker.task"
_HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_hand_model() -> Path:
    """Download hand_landmarker.task if not already cached."""
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not _HAND_MODEL_PATH.exists():
        logger.info("Downloading hand_landmarker.task model (~9 MB)...")
        print("Downloading hand_landmarker.task (first run only)...")
        urllib.request.urlretrieve(_HAND_MODEL_URL, _HAND_MODEL_PATH)
        logger.info("hand_landmarker.task downloaded to %s", _HAND_MODEL_PATH)
        print(f"[OK] Model saved to {_HAND_MODEL_PATH}")
    return _HAND_MODEL_PATH


# ---------------------------------------------------------------------------
# HandDetector — Tasks API wrapper
# ---------------------------------------------------------------------------

class HandDetector:
    """MediaPipe HandLandmarker wrapper (Tasks API).

    Responsible only for detection and drawing — no preprocessing logic.
    """

    # Landmark connections for drawing (indices match MediaPipe hand topology)
    _CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        min_presence_confidence: float = 0.7,
    ):
        model_path = _ensure_hand_model()

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._frame_ts_ms = 0  # monotonic timestamp for VIDEO mode
        logger.info(
            "HandDetector initialised (det=%.2f, trk=%.2f)",
            min_detection_confidence,
            min_tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> Any:
        """Run MediaPipe on a BGR frame.

        Returns a HandLandmarkerResult-like object with a
        ``multi_hand_landmarks`` attribute for compatibility with the
        rest of the pipeline.

        Increments an internal timestamp so VIDEO mode works correctly.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts_ms += 33  # ~30 FPS; never decreasing
        raw = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)
        return _ResultAdapter(raw)

    def draw_landmarks(self, frame_bgr: np.ndarray, hand_landmarks) -> None:
        """Draw landmarks + connections onto *frame_bgr* in-place using OpenCV.

        Uses cv2 directly — avoids Tasks API drawing_utils which requires
        additional landmark fields (.visibility) not present in our proxy.
        """
        h, w = frame_bgr.shape[:2]
        lms = hand_landmarks.landmark

        # Draw connections
        for connection in self._CONNECTIONS:
            s, e = connection.start, connection.end
            x1, y1 = int(lms[s].x * w), int(lms[s].y * h)
            x2, y2 = int(lms[e].x * w), int(lms[e].y * h)
            cv2.line(frame_bgr, (x1, y1), (x2, y2), (0, 220, 80), 2, cv2.LINE_AA)

        # Draw landmark joints
        for lm in lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame_bgr, (cx, cy), 5, (255, 255, 255), -1)
            cv2.circle(frame_bgr, (cx, cy), 5, (0, 140, 255), 2)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
        logger.info("HandDetector closed")


# ---------------------------------------------------------------------------
# Adapter: make Tasks API result look like the old solutions API
# ---------------------------------------------------------------------------

class _NormalizedLandmark:
    """Minimal landmark proxy with .x .y .z attributes."""
    __slots__ = ("x", "y", "z")

    def __init__(self, lm):
        self.x = lm.x
        self.y = lm.y
        self.z = lm.z


class _HandLandmarksProxy:
    """Proxy that exposes .landmark as a list of _NormalizedLandmark,
    compatible with normalize_landmarks() and draw_landmarks()."""

    def __init__(self, landmark_list):
        self.landmark = [_NormalizedLandmark(lm) for lm in landmark_list]


class _ResultAdapter:
    """Adapter that wraps HandLandmarkerResult to expose
    .multi_hand_landmarks as a list (matches old solutions API)."""

    def __init__(self, result):
        if result.hand_landmarks:
            self.multi_hand_landmarks = [
                _HandLandmarksProxy(lm_list)
                for lm_list in result.hand_landmarks
            ]
        else:
            self.multi_hand_landmarks = None


# ---------------------------------------------------------------------------
# Pure function — no class dependency
# ---------------------------------------------------------------------------

def normalize_landmarks(landmarks) -> np.ndarray:
    """Convert 21 MediaPipe landmarks into a (1, 63) feature vector.

    Steps:
      1. Subtract wrist (landmark 0) → relative coordinates
      2. Divide by max absolute value → scale invariant
      3. Flatten to shape (1, 63)

    Parameters
    ----------
    landmarks : list-like of objects with .x .y .z

    Returns
    -------
    np.ndarray of shape (1, 63)
    """
    landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    base = landmarks_np[0].copy()
    relative = landmarks_np - base
    max_val = np.max(np.abs(relative))
    if max_val > 0:
        normalized = relative / max_val
    else:
        normalized = relative
    return normalized.flatten().reshape(1, -1)
