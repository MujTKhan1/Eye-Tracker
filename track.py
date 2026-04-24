import argparse
import ctypes
import pickle
import time
from pathlib import Path

import cv2
import numpy as np

from src.gaze_core import build_face_mesh, draw_status, extract_features, get_screen_size, split_feature_vector


MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
CLICKABLE_CONTROL_TYPES = {
    "ButtonControl",
    "CheckBoxControl",
    "ComboBoxControl",
    "EditControl",
    "HyperlinkControl",
    "MenuItemControl",
    "RadioButtonControl",
    "TabItemControl",
}
DENSE_CLICKABLE_CONTROL_TYPES = {
    "DataItemControl",
    "ListItemControl",
    "TreeItemControl",
}


def predict_xy(feature_vec: np.ndarray, W: np.ndarray):
    x_aug = np.concatenate([feature_vec, np.array([1.0], dtype=np.float32)], axis=0)
    pred = x_aug @ W
    return float(pred[0]), float(pred[1])


def apply_axis_correction(x: float, y: float, corr: dict):
    if not corr:
        return x, y
    x = corr.get("x_a", 1.0) * x + corr.get("x_b", 0.0)
    y = corr.get("y_a", 1.0) * y + corr.get("y_b", 0.0)
    return float(x), float(y)


def clamp_xy(x: float, y: float, screen_w: int, screen_h: int):
    return int(np.clip(x, 0, screen_w - 1)), int(np.clip(y, 0, screen_h - 1))


def apply_edge_assist_1d(value: float, size: int, zone_ratio: float, max_push_px: float):
    zone = max(1.0, float(size) * float(zone_ratio))
    pushed = float(value)
    edge_active = False

    if pushed < zone:
        t = 1.0 - (pushed / zone)
        pushed -= float(max_push_px) * (t * t)
        edge_active = True
    elif pushed > float(size) - zone:
        t = (pushed - (float(size) - zone)) / zone
        pushed += float(max_push_px) * (t * t)
        edge_active = True

    return pushed, edge_active


def apply_edge_assist(x: float, y: float, screen_w: int, screen_h: int, zone_ratio: float, max_push_px: float):
    x2, edge_x = apply_edge_assist_1d(x, screen_w, zone_ratio, max_push_px)
    y2, edge_y = apply_edge_assist_1d(y, screen_h, zone_ratio, max_push_px)
    return x2, y2, (edge_x or edge_y)


class MouseController:
    def __init__(self):
        self.user32 = ctypes.windll.user32

    def move_to(self, x: int, y: int):
        self.user32.SetCursorPos(int(x), int(y))

    def left_click(self):
        self.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        self.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def double_click(self, interval: float = 0.06):
        self.left_click()
        time.sleep(float(interval))
        self.left_click()


class ClickableSnapper:
    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        snap_radius_px: float,
        hold_time: float,
        dwell_time: float,
        dwell_radius_px: float,
        unlock_distance_px: float,
        include_dense_items: bool,
        scan_step_px: int = 24,
    ):
        self.screen_w = int(screen_w)
        self.screen_h = int(screen_h)
        self.snap_radius_px = float(snap_radius_px)
        self.hold_time = float(hold_time)
        self.dwell_time = float(dwell_time)
        self.dwell_radius_px = float(dwell_radius_px)
        self.unlock_distance_px = float(unlock_distance_px)
        self.include_dense_items = bool(include_dense_items)
        self.scan_step_px = max(12, int(scan_step_px))
        self.max_area = float(screen_w * screen_h) * 0.18
        self.max_width = float(screen_w) * 0.9
        self.max_height = float(screen_h) * 0.45
        self.max_dense_area = float(screen_w * screen_h) * 0.035
        self.max_dense_width = float(screen_w) * 0.55
        self.max_dense_height = float(screen_h) * 0.18
        self.locked_target = None
        self.locked_until = 0.0
        self.locked_label = ""
        self.pending_anchor = None
        self.pending_started_at = 0.0
        self.enabled = False
        self.auto = None
        try:
            import uiautomation as auto

            self.auto = auto
            self.enabled = True
        except Exception:
            self.auto = None

    def reset(self):
        self.locked_target = None
        self.locked_until = 0.0
        self.locked_label = ""
        self.pending_anchor = None
        self.pending_started_at = 0.0

    def _control_key(self, control, rect):
        return (
            str(getattr(control, "ControlTypeName", "")),
            str(getattr(control, "Name", "")),
            int(rect.left),
            int(rect.top),
            int(rect.right),
            int(rect.bottom),
        )

    def _is_clickable(self, control, rect):
        if rect is None or rect.isempty():
            return False
        control_type = str(getattr(control, "ControlTypeName", ""))
        width = float(rect.width())
        height = float(rect.height())
        if width < 8 or height < 8:
            return False
        if not bool(getattr(control, "IsEnabled", True)):
            return False

        allowed = set(CLICKABLE_CONTROL_TYPES)
        if self.include_dense_items:
            allowed.update(DENSE_CLICKABLE_CONTROL_TYPES)
        if control_type not in allowed:
            return False

        if control_type in DENSE_CLICKABLE_CONTROL_TYPES:
            if width > self.max_dense_width or height > self.max_dense_height:
                return False
            if width * height > self.max_dense_area:
                return False
        else:
            if width > self.max_width or height > self.max_height:
                return False
            if width * height > self.max_area:
                return False

        return True

    def _candidate_from_control(self, control, raw_x: int, raw_y: int):
        current = control
        for _ in range(5):
            if current is None:
                break
            rect = getattr(current, "BoundingRectangle", None)
            if self._is_clickable(current, rect):
                cx = int(rect.xcenter())
                cy = int(rect.ycenter())
                dist = float(np.hypot(cx - raw_x, cy - raw_y))
                if dist <= self.snap_radius_px:
                    return {
                        "x": cx,
                        "y": cy,
                        "dist": dist,
                        "key": self._control_key(current, rect),
                        "label": f"{current.ControlTypeName}:{current.Name}" if current.Name else current.ControlTypeName,
                    }
            try:
                current = current.GetParentControl()
            except Exception:
                break
        return None

    def _scan_points(self, raw_x: int, raw_y: int):
        step = self.scan_step_px
        offsets = [
            (0, 0),
            (-step, 0),
            (step, 0),
            (0, -step),
            (0, step),
            (-step, -step),
            (step, -step),
            (-step, step),
            (step, step),
            (-2 * step, 0),
            (2 * step, 0),
            (0, -2 * step),
            (0, 2 * step),
        ]
        for dx, dy in offsets:
            yield (
                int(np.clip(raw_x + dx, 0, self.screen_w - 1)),
                int(np.clip(raw_y + dy, 0, self.screen_h - 1)),
            )

    def _find_candidate(self, raw_x: int, raw_y: int):
        if not self.enabled or self.auto is None:
            return None

        best = None
        seen = set()
        for px, py in self._scan_points(raw_x, raw_y):
            try:
                control = self.auto.ControlFromPoint(px, py)
            except Exception:
                continue
            candidate = self._candidate_from_control(control, raw_x, raw_y)
            if not candidate:
                continue
            if candidate["key"] in seen:
                continue
            seen.add(candidate["key"])
            if best is None or candidate["dist"] < best["dist"]:
                best = candidate
        return best

    def apply(self, raw_x: int, raw_y: int, request_snap: bool, now: float):
        if not self.enabled:
            return raw_x, raw_y, False, "uia-off"

        if self.locked_target is not None:
            lock_dist = float(np.hypot(raw_x - self.locked_target[0], raw_y - self.locked_target[1]))
            if lock_dist > self.unlock_distance_px:
                self.reset()

        if request_snap:
            if self.pending_anchor is None:
                self.pending_anchor = (raw_x, raw_y)
                self.pending_started_at = now
            else:
                anchor_dist = float(np.hypot(raw_x - self.pending_anchor[0], raw_y - self.pending_anchor[1]))
                if anchor_dist > self.dwell_radius_px:
                    self.pending_anchor = (raw_x, raw_y)
                    self.pending_started_at = now

            if self.locked_target is None and now - self.pending_started_at >= self.dwell_time:
                candidate = self._find_candidate(raw_x, raw_y)
                if candidate is not None:
                    self.locked_target = (candidate["x"], candidate["y"])
                    self.locked_until = now + self.hold_time
                    self.locked_label = candidate["label"]
                    self.pending_anchor = None
                    self.pending_started_at = 0.0
        else:
            self.pending_anchor = None
            self.pending_started_at = 0.0

        if self.locked_target is not None and now <= self.locked_until:
            return self.locked_target[0], self.locked_target[1], True, self.locked_label

        self.reset()
        return raw_x, raw_y, False, ""


class BlinkGuard:
    def __init__(self, freeze_threshold: float, recovery_time: float):
        self.freeze_threshold = float(freeze_threshold)
        self.recovery_time = float(recovery_time)
        self.currently_closed = False
        self.last_blink_ended_at = 0.0

    def reset(self):
        self.currently_closed = False
        self.last_blink_ended_at = 0.0

    def update(self, left_blink: float, right_blink: float, now: float):
        mean_blink = (float(left_blink) + float(right_blink)) / 2.0
        both_closed = mean_blink < self.freeze_threshold
        if both_closed and not self.currently_closed:
            self.currently_closed = True
        elif not both_closed and self.currently_closed:
            self.currently_closed = False
            self.last_blink_ended_at = now

    def should_freeze_motion(self, left_blink: float, right_blink: float, now: float):
        mean_blink = (float(left_blink) + float(right_blink)) / 2.0
        partially_closed = mean_blink < self.freeze_threshold
        in_recovery = self.last_blink_ended_at > 0.0 and now - self.last_blink_ended_at <= self.recovery_time
        return self.currently_closed or partially_closed or in_recovery


class MouthClickDetector:
    def __init__(
        self,
        open_threshold: float,
        release_threshold: float,
        single_open_time: float,
        double_open_time: float,
        click_cooldown: float,
    ):
        self.open_threshold = float(open_threshold)
        self.release_threshold = float(release_threshold)
        self.single_open_time = float(single_open_time)
        self.double_open_time = float(double_open_time)
        self.click_cooldown = float(click_cooldown)
        self.currently_open = False
        self.open_started_at = 0.0
        self.action_fired_this_open = False
        self.last_action_at = 0.0

    def reset(self):
        self.currently_open = False
        self.open_started_at = 0.0
        self.action_fired_this_open = False

    def update(self, mouth_open: float, now: float):
        mouth_open = float(mouth_open)
        action = None
        state = "closed"

        if not self.currently_open:
            if mouth_open >= self.open_threshold:
                self.currently_open = True
                self.open_started_at = now
                self.action_fired_this_open = False
                state = "opening"
        else:
            open_duration = now - self.open_started_at
            cooldown_ready = now - self.last_action_at >= self.click_cooldown

            if (
                not self.action_fired_this_open
                and open_duration >= self.double_open_time
                and cooldown_ready
            ):
                self.action_fired_this_open = True
                self.last_action_at = now
                action = "double"
                state = "double"
            elif mouth_open <= self.release_threshold:
                if (
                    not self.action_fired_this_open
                    and open_duration >= self.single_open_time
                    and cooldown_ready
                ):
                    self.last_action_at = now
                    action = "single"
                    state = "single"
                self.currently_open = False
                self.open_started_at = 0.0
                self.action_fired_this_open = False
                if action is None:
                    state = "closed"
            else:
                if self.action_fired_this_open:
                    state = "open"
                elif open_duration >= self.single_open_time:
                    state = "armed"
                else:
                    state = "opening"

        return action, state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=Path("calibration_model.pkl"))
    parser.add_argument("--smooth", type=float, default=0.25, help="0..1, higher is snappier")
    parser.add_argument("--head-assist", type=float, default=None, help="0..1 blend weight for head/face assist in hybrid mode")
    parser.add_argument("--y-gain", type=float, default=1.0, help="vertical sensitivity multiplier")
    parser.add_argument("--control-mouse", action="store_true", help="move the real Windows mouse cursor with gaze")
    parser.add_argument("--mouth-open-click", action="store_true", help="trigger a left click after opening your mouth briefly")
    parser.add_argument("--show-overlay", action="store_true", help="keep the fullscreen debug overlay visible in mouse-control mode")
    parser.add_argument("--cursor-deadzone-px", type=float, default=6.0, help="ignore tiny cursor moves under this pixel distance")
    parser.add_argument("--edge-zone-ratio", type=float, default=0.14, help="outer screen band ratio where edge assist starts pushing outward")
    parser.add_argument("--edge-boost-px", type=float, default=120.0, help="maximum outward push near the screen edges")
    parser.add_argument("--snap-clickables", action="store_true", help="snap to nearby clickable desktop controls while a mouth click is arming")
    parser.add_argument("--snap-radius-px", type=float, default=90.0, help="max distance for snapping to a nearby clickable control")
    parser.add_argument("--snap-hold", type=float, default=0.35, help="seconds to hold onto a snapped control after acquisition")
    parser.add_argument("--snap-dwell", type=float, default=0.12, help="seconds gaze must stay steady before snapping to a control")
    parser.add_argument("--snap-dwell-radius-px", type=float, default=28.0, help="allowed movement during snap dwell acquisition")
    parser.add_argument("--snap-unlock-distance-px", type=float, default=140.0, help="release a snapped target early if gaze moves this far away")
    parser.add_argument("--snap-browse-items", action="store_true", help="also allow list/tree/data items as snap targets in dense apps like File Explorer")
    parser.add_argument("--blink-freeze-threshold", type=float, default=0.195, help="freeze cursor when a blink compresses the eye ratio below this threshold")
    parser.add_argument("--blink-recovery", type=float, default=0.12, help="seconds to hold cursor steady after a blink ends")
    parser.add_argument("--mouth-open-threshold", type=float, default=0.20, help="mouth-open ratio required to arm a click")
    parser.add_argument("--mouth-release-threshold", type=float, default=0.12, help="mouth-open ratio required before another mouth click can arm")
    parser.add_argument("--mouth-single-open", type=float, default=0.10, help="mouth-open duration that becomes a single click when you close your mouth")
    parser.add_argument("--mouth-double-open", type=float, default=0.45, help="mouth-open duration that becomes a double click while still open")
    parser.add_argument("--click-cooldown", type=float, default=0.8, help="minimum seconds between mouth-triggered clicks")
    args = parser.parse_args()

    if args.mouth_release_threshold >= args.mouth_open_threshold:
        raise ValueError("--mouth-release-threshold must be lower than --mouth-open-threshold.")
    if args.mouth_double_open <= args.mouth_single_open:
        raise ValueError("--mouth-double-open must be greater than --mouth-single-open.")
    if not (0.0 < args.edge_zone_ratio < 0.5):
        raise ValueError("--edge-zone-ratio must be between 0 and 0.5.")

    if not args.model.exists():
        raise FileNotFoundError(f"Calibration model not found: {args.model}. Run calibrate.py first.")

    with args.model.open("rb") as f:
        payload = pickle.load(f)

    W = payload["W"]
    feature_dim = int(payload["feature_dim"])
    W_eye = payload.get("W_eye")
    W_head = payload.get("W_head")
    corr_full = payload.get("corr_full", {})
    corr_eye = payload.get("corr_eye", {})
    corr_head = payload.get("corr_head", {})
    eye_feature_dim = int(payload.get("eye_feature_dim", -1))
    head_feature_dim = int(payload.get("head_feature_dim", -1))
    default_head_assist = float(payload.get("head_assist_default", 0.0))
    head_assist = default_head_assist if args.head_assist is None else float(args.head_assist)
    head_assist = float(np.clip(head_assist, 0.0, 1.0))
    use_hybrid = W_eye is not None and W_head is not None
    model_screen_w, model_screen_h = payload["screen_size"]

    current_w, current_h = get_screen_size()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    face_mesh = build_face_mesh()
    mouse = MouseController() if args.control_mouse or args.mouth_open_click else None
    blink_guard = BlinkGuard(args.blink_freeze_threshold, args.blink_recovery)
    snapper = ClickableSnapper(
        current_w,
        current_h,
        snap_radius_px=args.snap_radius_px,
        hold_time=args.snap_hold,
        dwell_time=args.snap_dwell,
        dwell_radius_px=args.snap_dwell_radius_px,
        unlock_distance_px=args.snap_unlock_distance_px,
        include_dense_items=args.snap_browse_items,
    )
    mouth_detector = MouthClickDetector(
        open_threshold=args.mouth_open_threshold,
        release_threshold=args.mouth_release_threshold,
        single_open_time=args.mouth_single_open,
        double_open_time=args.mouth_double_open,
        click_cooldown=args.click_cooldown,
    )

    show_overlay = (not args.control_mouse) or args.show_overlay
    if show_overlay:
        cv2.namedWindow("Gaze Dot", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Gaze Dot", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    smooth_x = current_w // 2
    smooth_y = current_h // 2
    last_cursor_x = smooth_x
    last_cursor_y = smooth_y

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        feat = extract_features(frame, face_mesh)
        preview = frame.copy()
        canvas = np.full((current_h, current_w, 3), 255, dtype=np.uint8) if show_overlay else None
        click_action = None
        mouth_state = "idle"
        snapped_to_control = False
        snap_label = ""

        if feat is None:
            if args.control_mouse:
                blink_guard.reset()
            if args.snap_clickables:
                snapper.reset()
            if args.mouth_open_click:
                mouth_detector.reset()
            preview = draw_status(preview, "Face not detected", ok=False)
            if show_overlay:
                cv2.putText(canvas, "Face not detected", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        elif feat.vector.shape[0] != feature_dim:
            if args.control_mouse:
                blink_guard.reset()
            if args.snap_clickables:
                snapper.reset()
            if args.mouth_open_click:
                mouth_detector.reset()
            preview = draw_status(preview, "Feature mismatch: recalibrate", ok=False)
            if show_overlay:
                cv2.putText(canvas, "Feature mismatch: recalibrate", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            now_ts = time.time()
            freeze_motion = False
            if args.control_mouse:
                blink_guard.update(feat.left_blink, feat.right_blink, now_ts)
                freeze_motion = blink_guard.should_freeze_motion(feat.left_blink, feat.right_blink, now_ts)

            if args.mouth_open_click:
                click_action, mouth_state = mouth_detector.update(feat.mouth_open, now_ts)
                if mouse is not None and click_action == "single":
                    mouse.left_click()
                elif mouse is not None and click_action == "double":
                    mouse.double_click()

            if use_hybrid:
                _, eye_vec, head_vec = split_feature_vector(feat.vector)
                if eye_vec.shape[0] != eye_feature_dim or head_vec.shape[0] != head_feature_dim:
                    if args.control_mouse:
                        blink_guard.reset()
                    if args.mouth_open_click:
                        mouth_detector.reset()
                    preview = draw_status(preview, "Hybrid feature mismatch: recalibrate", ok=False)
                    if show_overlay:
                        cv2.putText(canvas, "Hybrid feature mismatch: recalibrate", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                    cv2.imshow("Webcam", preview)
                    if show_overlay:
                        cv2.imshow("Gaze Dot", canvas)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break
                    continue
                px_eye, py_eye = predict_xy(eye_vec, W_eye)
                px_head, py_head = predict_xy(head_vec, W_head)
                px_eye, py_eye = apply_axis_correction(px_eye, py_eye, corr_eye)
                px_head, py_head = apply_axis_correction(px_head, py_head, corr_head)
                px = (1.0 - head_assist) * px_eye + head_assist * px_head
                py = (1.0 - head_assist) * py_eye + head_assist * py_head
            else:
                px, py = predict_xy(feat.vector, W)
                px, py = apply_axis_correction(px, py, corr_full)

            px *= current_w / max(float(model_screen_w), 1.0)
            py *= current_h / max(float(model_screen_h), 1.0)
            py = (py - current_h / 2.0) * float(args.y_gain) + current_h / 2.0
            px, py, edge_assist_active = apply_edge_assist(
                px,
                py,
                current_w,
                current_h,
                zone_ratio=args.edge_zone_ratio,
                max_push_px=args.edge_boost_px,
            )

            px, py = clamp_xy(px, py, current_w, current_h)
            if not freeze_motion:
                smooth_x = int((1.0 - args.smooth) * smooth_x + args.smooth * px)
                smooth_y = int((1.0 - args.smooth) * smooth_y + args.smooth * py)

            request_snap = (
                args.snap_clickables
                and args.control_mouse
                and args.mouth_open_click
                and not freeze_motion
                and not edge_assist_active
                and mouth_state in {"armed", "open", "double"}
            )
            if args.snap_clickables and args.control_mouse:
                if edge_assist_active:
                    snapper.reset()
                smooth_x, smooth_y, snapped_to_control, snap_label = snapper.apply(
                    smooth_x,
                    smooth_y,
                    request_snap=request_snap,
                    now=now_ts,
                )

            mode_text = f"hybrid:{head_assist:.2f}" if use_hybrid else "eye-only"
            preview = draw_status(preview, f"Gaze: ({smooth_x}, {smooth_y}) {mode_text}")
            cv2.putText(
                preview,
                f"blink L:{feat.left_blink:.2f} R:{feat.right_blink:.2f} mouth:{feat.mouth_open:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (40, 40, 40),
                2,
            )
            if freeze_motion:
                cv2.putText(preview, "Blink hold", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 120, 220), 2)
            if args.mouth_open_click:
                mouth_color = (0, 180, 0) if click_action is not None else (50, 50, 50)
                cv2.putText(preview, f"Mouth click: {mouth_state}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mouth_color, 2)
            if edge_assist_active:
                cv2.putText(preview, "Edge assist", (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 90, 20), 2)
            if snapped_to_control:
                cv2.putText(preview, f"Snap: {snap_label}", (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 80, 20), 2)

            if args.control_mouse and mouse is not None:
                cursor_delta = float(np.hypot(smooth_x - last_cursor_x, smooth_y - last_cursor_y))
                if not freeze_motion and cursor_delta >= args.cursor_deadzone_px:
                    mouse.move_to(smooth_x, smooth_y)
                    last_cursor_x = smooth_x
                    last_cursor_y = smooth_y

            if show_overlay:
                cv2.circle(canvas, (smooth_x, smooth_y), 16, (0, 0, 255), -1)
                cv2.circle(canvas, (smooth_x, smooth_y), 30, (0, 100, 255), 2)

        if show_overlay:
            overlay_text = "Mouse mode ON" if args.control_mouse else "Mouse mode OFF"
            cv2.putText(canvas, overlay_text, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2)
            cv2.putText(canvas, "Press ESC to quit", (40, current_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)

        cv2.imshow("Webcam", preview)
        if show_overlay:
            cv2.imshow("Gaze Dot", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
