import ctypes
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class FrameFeatures:
    vector: np.ndarray
    frame: np.ndarray


def get_screen_size() -> Tuple[int, int]:
    user32 = ctypes.windll.user32
    return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))


def build_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _landmark_xy(landmarks, idx: int, width: int, height: int) -> Tuple[float, float]:
    p = landmarks[idx]
    return p.x * width, p.y * height


def _normalize_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    out = points.copy()
    out[:, 0] /= float(width)
    out[:, 1] /= float(height)
    return out.reshape(-1)


def extract_features(frame: np.ndarray, face_mesh) -> Optional[FrameFeatures]:
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark

    # Stable subset around eyes + key face anchors.
    indices = [
        33, 133, 159, 145,  # left eye
        362, 263, 386, 374,  # right eye
        468, 473,  # iris centers
        1, 4,  # nose bridge / tip
        61, 291,  # mouth corners (head orientation cue)
        10, 152,  # forehead/chin
    ]

    pts = np.array([_landmark_xy(lm, i, w, h) for i in indices], dtype=np.float32)

    # Eye geometry features.
    left_w = np.linalg.norm(pts[0] - pts[1]) + 1e-6
    right_w = np.linalg.norm(pts[4] - pts[5]) + 1e-6
    left_h = np.linalg.norm(pts[2] - pts[3])
    right_h = np.linalg.norm(pts[6] - pts[7])
    left_blink = left_h / left_w
    right_blink = right_h / right_w

    flat = _normalize_points(pts, w, h)
    extra = np.array([
        left_blink,
        right_blink,
        float(w) / max(float(h), 1.0),
    ], dtype=np.float32)

    return FrameFeatures(vector=np.concatenate([flat, extra]), frame=frame)


def draw_status(frame: np.ndarray, text: str, ok: bool = True) -> np.ndarray:
    color = (0, 220, 0) if ok else (0, 0, 255)
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return frame
