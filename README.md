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

## Tips for better accuracy
- Sit at normal viewing distance and keep posture steady during calibration.
- Use bright, even lighting.
- Re-run calibration if your monitor setup or seating position changes.
