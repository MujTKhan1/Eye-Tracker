import argparse
import pickle
import time
from pathlib import Path

import cv2
import numpy as np

from src.gaze_core import build_face_mesh, draw_status, extract_features, get_screen_size, split_feature_vector


def build_grid_points(screen_w: int, screen_h: int, cols: int, rows: int):
    xs = np.linspace(0.12, 0.88, cols)
    ys = np.linspace(0.12, 0.88, rows)
    return [(int(x * screen_w), int(y * screen_h)) for y in ys for x in xs]


def fit_linear_model(X: np.ndarray, Y: np.ndarray, ridge_lambda: float = 6.0):
    X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float32)])
    # Ridge-regularized linear regression (do not regularize bias term).
    xtx = X_aug.T @ X_aug
    reg = np.eye(xtx.shape[0], dtype=np.float32) * float(ridge_lambda)
    reg[-1, -1] = 0.0
    xty = X_aug.T @ Y
    W = np.linalg.solve(xtx + reg, xty)
    return W.astype(np.float32)


def fit_axis_correction(Y_pred: np.ndarray, Y_true: np.ndarray):
    # Fit per-axis affine correction: y_true = a * y_pred + b
    out = {}
    for axis, key in enumerate(("x", "y")):
        p = Y_pred[:, axis]
        t = Y_true[:, axis]
        A = np.vstack([p, np.ones_like(p)]).T
        coeff, _, _, _ = np.linalg.lstsq(A, t, rcond=None)
        out[f"{key}_a"] = float(coeff[0])
        out[f"{key}_b"] = float(coeff[1])
    return out


def filter_point_samples(samples: list, keep_ratio: float = 0.7):
    arr = np.asarray(samples, dtype=np.float32)
    if len(arr) <= 3:
        return arr
    center = np.median(arr, axis=0)
    dist = np.linalg.norm(arr - center, axis=1)
    threshold = float(np.quantile(dist, keep_ratio))
    mask = dist <= threshold
    kept = arr[mask]
    if len(kept) < 3:
        return arr
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--samples-per-point", type=int, default=25)
    parser.add_argument("--grid", type=str, default="4x3", help="format: colsxrows")
    parser.add_argument("--output", type=Path, default=Path("calibration_model.pkl"))
    parser.add_argument("--ridge", type=float, default=6.0, help="regularization strength for calibration fit")
    parser.add_argument("--keep-ratio", type=float, default=0.7, help="per-point sample retention ratio after outlier filtering")
    args = parser.parse_args()

    cols, rows = [int(v) for v in args.grid.lower().split("x")]
    screen_w, screen_h = get_screen_size()
    targets = build_grid_points(screen_w, screen_h, cols, rows)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    face_mesh = build_face_mesh()

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    X, X_eye, X_head, Y = [], [], [], []

    intro = np.full((screen_h, screen_w, 3), 245, dtype=np.uint8)
    cv2.putText(intro, "Calibration: Look at each red dot", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    cv2.putText(intro, "Press SPACE to begin. Press ESC anytime to cancel.", (80, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.imshow("Calibration", intro)
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return
        if key == 32:
            break

    for idx, target in enumerate(targets, start=1):
        point_samples = []
        # Settling phase.
        t0 = time.time()
        while time.time() - t0 < 0.8:
            canvas = np.full((screen_h, screen_w, 3), 245, dtype=np.uint8)
            cv2.circle(canvas, target, 20, (0, 0, 255), -1)
            cv2.putText(canvas, f"Point {idx}/{len(targets)}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
            cv2.putText(canvas, "Hold your gaze on the dot", (40, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
            cv2.imshow("Calibration", canvas)
            if (cv2.waitKey(1) & 0xFF) == 27:
                cap.release()
                cv2.destroyAllWindows()
                return

        collected = 0
        while collected < args.samples_per_point:
            ok, frame = cap.read()
            if not ok:
                continue

            feat = extract_features(frame, face_mesh)
            preview = frame.copy()
            if feat is None:
                preview = draw_status(preview, "Face not detected", ok=False)
            else:
                point_samples.append(feat.vector)
                collected += 1
                preview = draw_status(preview, f"Captured {collected}/{args.samples_per_point}")

            cv2.imshow("Webcam", preview)

            canvas = np.full((screen_h, screen_w, 3), 245, dtype=np.uint8)
            cv2.circle(canvas, target, 20, (0, 0, 255), -1)
            cv2.putText(canvas, f"Point {idx}/{len(targets)}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
            cv2.putText(canvas, f"Samples {collected}/{args.samples_per_point}", (40, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
            cv2.imshow("Calibration", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return

        cleaned = filter_point_samples(point_samples, keep_ratio=float(args.keep_ratio))
        for sample in cleaned:
            full_vec, eye_vec, head_vec = split_feature_vector(sample)
            X.append(full_vec)
            X_eye.append(eye_vec)
            X_head.append(head_vec)
            Y.append(target)

    X_np = np.asarray(X, dtype=np.float32)
    X_eye_np = np.asarray(X_eye, dtype=np.float32)
    X_head_np = np.asarray(X_head, dtype=np.float32)
    Y_np = np.asarray(Y, dtype=np.float32)

    if len(X_np) < 20:
        raise RuntimeError("Not enough calibration samples. Try again.")

    W = fit_linear_model(X_np, Y_np, ridge_lambda=float(args.ridge))
    W_eye = fit_linear_model(X_eye_np, Y_np, ridge_lambda=float(args.ridge))
    W_head = fit_linear_model(X_head_np, Y_np, ridge_lambda=float(args.ridge) * 0.8)

    X_aug = np.hstack([X_np, np.ones((X_np.shape[0], 1), dtype=np.float32)])
    X_eye_aug = np.hstack([X_eye_np, np.ones((X_eye_np.shape[0], 1), dtype=np.float32)])
    X_head_aug = np.hstack([X_head_np, np.ones((X_head_np.shape[0], 1), dtype=np.float32)])
    corr_full = fit_axis_correction(X_aug @ W, Y_np)
    corr_eye = fit_axis_correction(X_eye_aug @ W_eye, Y_np)
    corr_head = fit_axis_correction(X_head_aug @ W_head, Y_np)

    payload = {
        "W": W,
        "W_eye": W_eye,
        "W_head": W_head,
        "corr_full": corr_full,
        "corr_eye": corr_eye,
        "corr_head": corr_head,
        "screen_size": (screen_w, screen_h),
        "feature_dim": int(X_np.shape[1]),
        "eye_feature_dim": int(X_eye_np.shape[1]),
        "head_feature_dim": int(X_head_np.shape[1]),
        "head_assist_default": 0.35,
        "grid": (cols, rows),
        "samples_per_point": args.samples_per_point,
        "ridge": float(args.ridge),
        "keep_ratio": float(args.keep_ratio),
    }

    with args.output.open("wb") as f:
        pickle.dump(payload, f)

    done = np.full((screen_h, screen_w, 3), 245, dtype=np.uint8)
    cv2.putText(done, f"Saved calibration to {args.output}", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 120, 20), 2)
    cv2.putText(done, "Press any key to close", (80, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.imshow("Calibration", done)
    cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
