import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

from src.gaze_core import build_face_mesh, draw_status, extract_features, get_screen_size


def predict_xy(feature_vec: np.ndarray, W: np.ndarray):
    x_aug = np.concatenate([feature_vec, np.array([1.0], dtype=np.float32)], axis=0)
    pred = x_aug @ W
    return float(pred[0]), float(pred[1])


def clamp_xy(x: float, y: float, screen_w: int, screen_h: int):
    return int(np.clip(x, 0, screen_w - 1)), int(np.clip(y, 0, screen_h - 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=Path("calibration_model.pkl"))
    parser.add_argument("--smooth", type=float, default=0.25, help="0..1, higher is snappier")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Calibration model not found: {args.model}. Run calibrate.py first.")

    with args.model.open("rb") as f:
        payload = pickle.load(f)

    W = payload["W"]
    feature_dim = int(payload["feature_dim"])
    model_screen_w, model_screen_h = payload["screen_size"]

    # If monitor changed since calibration, we still render on current screen.
    current_w, current_h = get_screen_size()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    face_mesh = build_face_mesh()

    cv2.namedWindow("Gaze Dot", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Gaze Dot", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    smooth_x = current_w // 2
    smooth_y = current_h // 2

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        feat = extract_features(frame, face_mesh)
        preview = frame.copy()
        canvas = np.full((current_h, current_w, 3), 255, dtype=np.uint8)

        if feat is None:
            preview = draw_status(preview, "Face not detected", ok=False)
            cv2.putText(canvas, "Face not detected", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        elif feat.vector.shape[0] != feature_dim:
            preview = draw_status(preview, "Feature mismatch: recalibrate", ok=False)
            cv2.putText(canvas, "Feature mismatch: recalibrate", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            px, py = predict_xy(feat.vector, W)

            # Scale prediction if display resolution changed after calibration.
            px *= current_w / max(float(model_screen_w), 1.0)
            py *= current_h / max(float(model_screen_h), 1.0)

            px, py = clamp_xy(px, py, current_w, current_h)
            smooth_x = int((1.0 - args.smooth) * smooth_x + args.smooth * px)
            smooth_y = int((1.0 - args.smooth) * smooth_y + args.smooth * py)

            preview = draw_status(preview, f"Gaze: ({smooth_x}, {smooth_y})")
            cv2.circle(canvas, (smooth_x, smooth_y), 16, (0, 0, 255), -1)
            cv2.circle(canvas, (smooth_x, smooth_y), 30, (0, 100, 255), 2)

        cv2.putText(canvas, "Press ESC to quit", (40, current_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)

        cv2.imshow("Webcam", preview)
        cv2.imshow("Gaze Dot", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
