# Webcam Eye Tracker (Calibration + Live Tracking)

This project tracks where you are looking on the screen using your webcam.

## What you get
- `calibrate.py`: collects gaze samples while you look at dots on screen
- `track.py`: predicts gaze in real-time and shows a red dot where you are looking

## Setup (Windows)

Use Python 3.9 for best MediaPipe compatibility.

```powershell
cd "C:\Users\pc\Desktop\Codex Workspace\webcam-eye-tracker"
py -3.9 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## 1) Calibrate

```powershell
.\.venv\Scripts\python calibrate.py
```

Controls:
- `SPACE`: start calibration
- `ESC`: cancel/exit

This saves `calibration_model.pkl`.

## 2) Track

```powershell
.\.venv\Scripts\python track.py
```

Controls:
- `ESC`: quit

Hybrid tuning:
- `--head-assist 0.0` = pure eye model
- `--head-assist 0.3` to `0.5` = eye + face/head assist (usually smoother)
- `--head-assist 1.0` = pure head/face model

Example:
```powershell
.\.venv\Scripts\python track.py --smooth 0.28 --head-assist 0.35
```

Mouse control example:
```powershell
.\.venv\Scripts\python track.py --control-mouse --mouth-open-click --snap-clickables --smooth 0.28 --head-assist 0.35 --y-gain 1.25
```

Mouse control flags:
- `--control-mouse` moves the real Windows cursor to the predicted gaze point
- `--mouth-open-click` uses mouth duration for clicks: short open = single click, long open = double click
- `--snap-clickables` latches onto nearby Windows buttons and fields while a mouth click is arming
- `--snap-browse-items` also allows list/tree/data items to snap in dense apps like File Explorer
- `--edge-zone-ratio` and `--edge-boost-px` help the cursor keep reaching true screen edges even when snapping shortens the range
- `--show-overlay` keeps the fullscreen gaze overlay visible while controlling the real mouse

## Tips for better accuracy
- Sit at normal viewing distance and keep posture steady during calibration.
- Use bright, even lighting.
- Re-run calibration if your monitor setup or seating position changes.
