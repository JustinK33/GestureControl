# GestureControl

A real-time computer vision app that detects faces and hand landmarks from a webcam and maps hand gestures to mouse control

---

## Tech Stack

- Python
- OpenCV
- MediaPipe Tasks Vision API
- PyAutoGUI
- ONNX (YuNet Face Detection)

---

## Features (updating...)

- **Real-time vision pipeline**
  - Captures live webcam feed and processes frames continuously
  - Detects and renders:
    - Multiple faces with bounding boxes
    - Multiple hands with 21-point landmarks and connections

- **Gesture-based interaction**
  - Control mouse cursor using **index fingertip tracking**
  - Perform **pinch-to-click gestures** using thumb + index finger
  - Adaptive click threshold based on hand size for improved accuracy

- **Multi-hand tracking**
  - Supports detection of up to 4 hands simultaneously
  - Renders full hand skeleton with landmark connections

---

## 📚 What I Learned From This Project

- **Real-time computer vision pipelines**
  Built a frame-by-frame pipeline for **capture → inference → rendering → interaction**, ensuring smooth real-time performance

- **Working with MediaPipe Tasks API**
  Learned how to use **LIVE_STREAM mode with async callbacks** for efficient hand landmark detection

- **Gesture recognition logic**
  Implemented gesture detection using **landmark geometry** and **adaptive thresholds** instead of hardcoded values

- **Coordinate transformations**
  Converted normalized landmark coordinates into **screen-space positions** for accurate cursor control

- **Human-computer interaction (HCI)**
  Designed intuitive controls (mirrored movement + pinch gestures) for natural webcam-based interaction

- **Debouncing and stability**
  Added **cooldowns and click debouncing** to prevent unintended rapid-fire inputs

---

## Running the Project

### To run the project locally, follow these steps:

1. Clone the repo  
   ```bash
   git clone <url>

2. Create a virtual environment
  python3 -m venv venv

3. Activate the environment
  source venv/bin/activate

4. Install requirements
  pip install -r requirements.txt

5. Run the app
  python main.py
