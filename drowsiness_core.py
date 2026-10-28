"""
Shared drowsiness detection engine aligned with FYP I (OpenCV + optional Dlib EAR).
Uses Haar cascades when Dlib landmarks are unavailable.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import distance as dist

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ALARM_PATH = PROJECT_ROOT / "alarm.wav"
DLIB_MODEL_PATH = PROJECT_ROOT / "models" / "shape_predictor_68_face_landmarks.dat"
FACE_CASCADE_PATH = PROJECT_ROOT / "models" / "haarcascade_frontalface_default.xml"

EAR_THRESHOLD = 0.27
MAR_YAWN_THRESHOLD = 0.55
CONSEC_CLOSED_FRAMES = 18
YAWN_CONSEC_FRAMES = 12

LEFT_EYE_IDX = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]
MOUTH_IDX = [48, 54, 51, 57]  # left, right, upper, lower inner lip


def euclidean(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return float(dist.euclidean(p1, p2))


def eye_aspect_ratio(eye: List[Tuple[int, int]]) -> float:
    a = euclidean(eye[1], eye[5])
    b = euclidean(eye[2], eye[4])
    c = euclidean(eye[0], eye[3])
    if c <= 1e-6:
        return 0.0
    return (a + b) / (2.0 * c)


def mouth_aspect_ratio(landmarks: List[Tuple[int, int]]) -> float:
    left, right, upper, lower = (landmarks[i] for i in MOUTH_IDX)
    vertical = euclidean(upper, lower)
    horizontal = euclidean(left, right)
    if horizontal <= 1e-6:
        return 0.0
    return vertical / horizontal


@dataclass
class LiveMetrics:
    ear: float = 0.0
    mar: float = 0.0
    status: str = "Initializing"
    blink_count: int = 0
    alert_count: int = 0
    closed_frames: int = 0
    yawn_frames: int = 0
    face_detected: bool = False
    backend: str = "opencv"
    last_updated: Optional[str] = None


class _OpenCvBackend:
    def __init__(self) -> None:
        face_path = str(FACE_CASCADE_PATH)
        self.face_cascade = cv2.CascadeClassifier(face_path)
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load face cascade: {face_path}")

    def close(self) -> None:
        pass

    def _pseudo_ear(self, eye_roi: np.ndarray) -> float:
        h, w = eye_roi.shape[:2]
        if h < 4 or w < 4:
            return 0.0
        gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY) if eye_roi.ndim == 3 else eye_roi
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        white = float(cv2.countNonZero(thresh))
        ratio = white / float(h * w)
        return min(0.45, ratio * 2.2)

    def process(
        self, frame: np.ndarray, state: "_EngineState"
    ) -> Tuple[np.ndarray, float, float, bool]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(120, 120))
        if len(faces) == 0:
            return frame, 0.0, 0.0, False

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 128, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 8, minSize=(24, 24))

        ears: List[float] = []
        for (ex, ey, ew, eh) in eyes[:2]:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 2)
            eye_patch = roi_color[ey : ey + eh, ex : ex + ew]
            ears.append(self._pseudo_ear(eye_patch))

        ear = float(np.mean(ears)) if ears else 0.0
        mar = 0.0
        mouth_y1 = y + int(h * 0.62)
        mouth_y2 = y + int(h * 0.92)
        mouth_x1 = x + int(w * 0.25)
        mouth_x2 = x + int(w * 0.75)
        mouth_roi = gray[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
        if mouth_roi.size > 0:
            mar = min(1.0, float(np.std(mouth_roi)) / 40.0)

        return frame, ear, mar, True


class _DlibBackend:
    FACE_DOWNSAMPLE = 1.5

    def __init__(self) -> None:
        import dlib

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(str(DLIB_MODEL_PATH))

    def close(self) -> None:
        pass

    def _landmarks(self, gray: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        import dlib

        small = cv2.resize(
            gray,
            None,
            fx=1.0 / self.FACE_DOWNSAMPLE,
            fy=1.0 / self.FACE_DOWNSAMPLE,
            interpolation=cv2.INTER_LINEAR,
        )
        rects = self.detector(small, 0)
        if not rects:
            return None
        rect = dlib.rectangle(
            int(rects[0].left() * self.FACE_DOWNSAMPLE),
            int(rects[0].top() * self.FACE_DOWNSAMPLE),
            int(rects[0].right() * self.FACE_DOWNSAMPLE),
            int(rects[0].bottom() * self.FACE_DOWNSAMPLE),
        )
        return [(p.x, p.y) for p in self.predictor(gray, rect).parts()]

    def process(
        self, frame: np.ndarray, state: "_EngineState"
    ) -> Tuple[np.ndarray, float, float, bool]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        landmarks = self._landmarks(gray)
        if not landmarks:
            return frame, 0.0, 0.0, False

        left = [landmarks[i] for i in LEFT_EYE_IDX]
        right = [landmarks[i] for i in RIGHT_EYE_IDX]
        ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
        mar = mouth_aspect_ratio(landmarks)

        for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
            cv2.circle(frame, landmarks[idx], 2, (0, 255, 255), -1)

        return frame, ear, mar, True


@dataclass
class _EngineState:
    closed_counter: int = 0
    yawn_counter: int = 0
    was_closed: bool = False


@dataclass
class DrowsinessEngine:
    ear_threshold: float = EAR_THRESHOLD
    mar_yawn_threshold: float = MAR_YAWN_THRESHOLD
    consec_closed_frames: int = CONSEC_CLOSED_FRAMES
    yawn_consec_frames: int = YAWN_CONSEC_FRAMES
    metrics: LiveMetrics = field(default_factory=LiveMetrics)
    _state: _EngineState = field(default_factory=_EngineState, init=False, repr=False)
    _backend: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._backend = self._create_backend()
        self.metrics.backend = getattr(self._backend, "__class__", type(self._backend)).__name__.replace(
            "_", ""
        ).replace("Backend", "").lower()
        if isinstance(self._backend, _DlibBackend):
            self.metrics.backend = "dlib"
        else:
            self.metrics.backend = "opencv"

    def _create_backend(self) -> object:
        if DLIB_MODEL_PATH.is_file():
            try:
                return _DlibBackend()
            except ImportError:
                pass
        return _OpenCvBackend()

    def process_bgr_frame(self, frame: np.ndarray) -> np.ndarray:
        frame, ear, mar, face_ok = self._backend.process(frame, self._state)
        self.metrics.face_detected = face_ok
        self.metrics.ear = ear
        self.metrics.mar = mar

        drowsy = False
        if face_ok:
            if ear < self.ear_threshold:
                self._state.closed_counter += 1
                if not self._state.was_closed:
                    self._state.was_closed = True
            else:
                if self._state.was_closed and self._state.closed_counter < self.consec_closed_frames:
                    self.metrics.blink_count += 1
                self._state.closed_counter = 0
                self._state.was_closed = False

            if mar >= self.mar_yawn_threshold:
                self._state.yawn_counter += 1
            else:
                self._state.yawn_counter = 0

            self.metrics.closed_frames = self._state.closed_counter
            self.metrics.yawn_frames = self._state.yawn_counter

            if self._state.closed_counter >= self.consec_closed_frames:
                drowsy = True
            if self._state.yawn_counter >= self.yawn_consec_frames:
                drowsy = True

        h, w = frame.shape[:2]
        if drowsy:
            self.metrics.status = "DROWSY"
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 8)
            cv2.putText(
                frame,
                "DROWSINESS ALERT!",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )
        elif face_ok:
            self.metrics.status = "ALERT"
            cv2.putText(
                frame,
                f"EAR: {ear:.2f} | Blinks: {self.metrics.blink_count} | Backend: {self.metrics.backend}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            self.metrics.status = "No face detected"
            cv2.putText(
                frame,
                "Face not detected — adjust lighting or camera position",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
            )

        self.metrics.last_updated = datetime.now().strftime("%H:%M:%S")
        return frame

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()


_alarm_lock = threading.Lock()
_alarm_playing = False


def play_alarm_async(alarm_path: Optional[Path] = None) -> None:
    global _alarm_playing
    path = Path(alarm_path) if alarm_path else DEFAULT_ALARM_PATH
    if not path.is_file():
        return

    with _alarm_lock:
        if _alarm_playing:
            return
        _alarm_playing = True

    def _play() -> None:
        global _alarm_playing
        try:
            import sys

            if sys.platform == "win32":
                import winsound

                winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                from playsound import playsound

                playsound(str(path))
        except Exception:
            pass
        finally:
            with _alarm_lock:
                _alarm_playing = False

    threading.Thread(target=_play, daemon=True).start()


def reset_alarm_gate() -> None:
    global _alarm_playing
    with _alarm_lock:
        _alarm_playing = False
