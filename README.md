# 🚗 FocusGuard — Real-Time Driver Drowsiness & Distraction Detector

> Inter-College Hackathon Project | Computer Vision + AI Safety | 2-Person Team

---

## What It Does

FocusGuard monitors a driver's face through any phone camera or webcam and detects drowsiness, distraction, and phone use — firing progressive alerts before an accident happens. No special hardware. No internet. No subscription.

---

## Quick Start (3 steps)

### Step 1 — Install dependencies

```bash
# Clone / download the project folder, then:
cd focusguard

# Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt

# Install React dependencies
cd dashboard && npm install && cd ..
```

### Step 2 — Start backend server

```bash
# Terminal 1
source venv/bin/activate
python server/app.py
# → Running at http://localhost:8000
```

### Step 3 — Start the dashboard

```bash
# Terminal 2
cd dashboard
npm start
# → Opens http://localhost:3000
```

---

## Running the Vision Engine (real camera)

```bash
# Terminal 3 — with webcam / phone camera
source venv/bin/activate
python vision/main.py

# Options:
python vision/main.py --camera 1          # use second camera
python vision/main.py --sensitivity high  # more sensitive alerts
python vision/main.py --no-display        # headless (no OpenCV window)
```

---

## Demo Mode (no camera needed — for hackathon)

```bash
# Make sure server is running first (Step 2), then:
source venv/bin/activate
python scripts/demo_mode.py

# Speed options:
python scripts/demo_mode.py --speed slow    # easier to follow
python scripts/demo_mode.py --speed fast    # quick demo
python scripts/demo_mode.py --loop          # loops continuously
```

Demo simulates 5 phases over ~2 minutes:
1. 🟢 Normal driving
2. 🟡 Early fatigue (EAR dropping)
3. 🟠 Drowsy (head nodding)
4. 🔴 Critical (alarm fires)
5. 🟢 Recovery

---

## One-Command Launch

```bash
# Starts everything: server + vision engine
python run.py

# Demo mode (no camera):
python run.py --demo

# With options:
python run.py --camera 1 --sensitivity high
```

---

## Project Structure

```
focusguard/
├── vision/
│   ├── main.py            ← Camera loop + orchestration
│   ├── face_detector.py   ← MediaPipe FaceMesh
│   ├── ear_calculator.py  ← EAR + PERCLOS algorithm
│   ├── head_pose.py       ← 3D head pose estimation
│   ├── yawn_detector.py   ← MAR-based yawn detection
│   ├── phone_detector.py  ← MobileNet phone detection
│   ├── alert_engine.py    ← Multi-level audio alerts
│   └── bridge.py          ← Vision→Server HTTP bridge
├── server/
│   ├── app.py             ← FastAPI + WebSocket server
│   └── database.py        ← SQLite incident logger
├── dashboard/
│   └── src/App.js         ← React live dashboard
├── reports/
│   └── report_generator.py← PDF session report
├── scripts/
│   ├── demo_mode.py       ← Hackathon demo simulator
│   └── setup.sh           ← One-command installer
├── data/                  ← SQLite database (auto-created)
├── sounds/                ← Alert WAV files (auto-generated)
├── models/                ← MobileNet SSD (optional)
├── run.py                 ← Single launcher
└── requirements.txt
```

---

## Alert Levels

| Level | Trigger | Alert |
|-------|---------|-------|
| 0 — Normal | EAR > 0.25, eyes forward | Green dashboard |
| 1 — Warning | EAR 0.20–0.25 for 1.5s | Soft beep |
| 2 — Danger | EAR < 0.20 for 2s or phone detected | Loud alarm + red flash |
| 3 — Critical | PERCLOS > 15% or eyes shut 4s | Continuous alarm + "PULL OVER" |

---

## How EAR Works

```
EAR = (||P2-P6|| + ||P3-P5||) / (2 × ||P1-P4||)

Eye open  → EAR ≈ 0.28–0.32
Drooping  → EAR ≈ 0.20–0.25  (Warning)
Closed    → EAR < 0.20        (Alert)
```

PERCLOS = % of last 60 seconds where EAR < 0.20.
Clinical fatigue threshold: PERCLOS > 15%.

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Face landmarks | MediaPipe FaceMesh (468 points) |
| CV processing | OpenCV 4.9 |
| ML inference | PyTorch / MobileNet SSD |
| Backend | FastAPI + WebSocket |
| Database | SQLite |
| Frontend | React 18 + Chart.js |
| Reports | ReportLab PDF |
| Audio alerts | pygame.mixer |

---

## Hardware Needed

**Minimum (₹0 cost):**
- Laptop with built-in webcam
- Built-in speakers or earphones

**Optional for in-car demo:**
- Phone dashboard mount (₹150–300)
- USB webcam Logitech C270 (₹1,200)

---

## Troubleshooting

**"No face detected" on dashboard:**
→ Check camera index: `python vision/main.py --camera 1`
→ Ensure good lighting on your face

**Dashboard shows "Disconnected":**
→ Make sure `python server/app.py` is running first
→ Check that port 8000 is not blocked

**No alert sounds:**
→ sounds/ folder is auto-created on first run
→ Check system volume / pygame install: `pip install pygame`

**Demo mode not updating dashboard:**
→ Start server first, then run demo_mode.py
→ Check API at http://localhost:8000/api/state

---

## Team

Built for inter-college hackathon — Open Innovation track.
2-person team | 12-day sprint | Computer Vision + Full-Stack

---

*FocusGuard — The co-pilot that never gets tired.*
