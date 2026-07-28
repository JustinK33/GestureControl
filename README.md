# GestureControl

A real-time computer vision app that tracks your hand and face from a webcam and lets you control your mouse with gestures.

<!-- TODO: add a demo gif of gesture control in action -->

## What It Does

Point your webcam at yourself and GestureControl tracks your face and both hands in real time, rendering face boxes and 21-point hand skeletons over the live feed.
Move your index finger to move the mouse cursor, and pinch your thumb and index finger together to click - the click threshold adapts to your hand size so it works up close or further from the camera.
It tracks up to 4 hands at once and debounces clicks so gestures don't fire repeatedly by accident.

## Tech Stack

- Python
- OpenCV
- MediaPipe Tasks Vision API
- PyAutoGUI
- ONNX (YuNet Face Detection)

## Install and Run

```bash
git clone <url>
cd GestureControl
conda create -n gesturecontrol python=3.10 -y
conda activate gesturecontrol
pip install -r requirements.txt
python main.py
```

## What I Learned

- **Real-time computer vision pipelines** - built a frame-by-frame pipeline for capture, inference, rendering, and interaction, keeping smooth real-time performance.
- **MediaPipe Tasks API** - used LIVE_STREAM mode with async callbacks for efficient hand landmark detection.
- **Gesture recognition logic** - detected gestures using landmark geometry and adaptive thresholds instead of hardcoded values.
- **Coordinate transformations** - converted normalized landmark coordinates into screen-space positions for accurate cursor control.
- **Human-computer interaction (HCI)** - designed intuitive controls (mirrored movement + pinch gestures) for natural webcam-based interaction.
- **Debouncing and stability** - added cooldowns and click debouncing to prevent unintended rapid-fire inputs.
