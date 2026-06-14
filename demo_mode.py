"""
demo_mode.py — Simulates a progressive drowsiness scenario for demo/testing.
Sends synthetic state dicts to the FastAPI /api/ingest endpoint.
No camera or real hardware needed.
"""

import requests
import time
import math
import argparse
import sys
import os

# Enable UTF-8 output on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

API_URL = "http://localhost:8000/api/ingest"

# Speed multipliers
SPEEDS = {"slow": 2.0, "normal": 1.0, "fast": 0.4}


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def build_state(phase: int, t: float) -> dict:
    """
    Build a synthetic state dict for a given phase and progress t (0.0–1.0).

    Phase 0: Normal driving (30s)
    Phase 1: Early fatigue (30s)
    Phase 2: Drowsy (30s)
    Phase 3: Critical (30s)
    Phase 4: Recovery (20s)
    """
    base = {
        "face_detected": True,
        "is_yawning":    False,
        "yawn_count":    0,
        "phone_detected": False,
        "head_pitch":    0.0,
        "head_yaw":      0.0,
        "head_roll":     0.0,
        "head_state":    "forward",
        "is_distracted": False,
        "fps":           28.0,
        "timestamp":     time.time(),
    }

    if phase == 0:
        # Normal — stable EAR, PERCLOS low
        ear = lerp(0.30, 0.28, t) + 0.005 * math.sin(t * 10)
        base.update({
            "ear_left":  ear + 0.01, "ear_right": ear - 0.01,
            "ear_avg":   ear, "perclos": lerp(2.0, 4.0, t),
            "alert_level": 0, "alert_reason": "Normal",
            "drowsiness_score": 3,
        })

    elif phase == 1:
        # Early fatigue — EAR starting to drop, occasional blink delay
        ear = lerp(0.27, 0.22, t) + 0.008 * math.sin(t * 6)
        perclos = lerp(5.0, 11.0, t)
        base.update({
            "ear_left":  ear + 0.008, "ear_right": ear - 0.008,
            "ear_avg":   ear, "perclos": perclos,
            "is_yawning": t > 0.6,
            "yawn_count": 1 if t > 0.7 else 0,
            "alert_level": 1 if ear < 0.24 else 0,
            "alert_reason": "Eyes drooping" if ear < 0.24 else "Normal",
            "drowsiness_score": int(perclos * 6.67),
        })

    elif phase == 2:
        # Drowsy — EAR below threshold, head starts nodding
        ear = lerp(0.21, 0.16, t)
        pitch = lerp(2.0, 14.0, t)
        perclos = lerp(12.0, 18.0, t)
        base.update({
            "ear_left":    ear + 0.005, "ear_right": ear - 0.005,
            "ear_avg":     ear, "perclos": perclos,
            "head_pitch":  pitch,
            "head_state":  "head_down" if pitch > 12 else "forward",
            "is_distracted": pitch > 12,
            "yawn_count":  2,
            "alert_level": 2 if ear < 0.18 else 1,
            "alert_reason": f"Eyes closed EAR={ear:.2f}",
            "drowsiness_score": min(100, int(perclos * 6.67)),
        })

    elif phase == 3:
        # Critical — eyes nearly shut, head down, PERCLOS critical
        ear = lerp(0.14, 0.09, t)
        pitch = lerp(15.0, 22.0, t)
        perclos = lerp(19.0, 26.0, t)
        base.update({
            "ear_left":    ear + 0.003, "ear_right": ear - 0.003,
            "ear_avg":     ear, "perclos": perclos,
            "head_pitch":  pitch,
            "head_state":  "head_down",
            "is_distracted": True,
            "yawn_count":  4,
            "alert_level": 3,
            "alert_reason": f"CRITICAL — EAR={ear:.2f} PERCLOS={perclos:.1f}%",
            "drowsiness_score": min(100, int(perclos * 6.67)),
        })

    elif phase == 4:
        # Recovery — driver wakes up after alarm
        ear = lerp(0.10, 0.29, t)
        perclos = lerp(24.0, 6.0, t)
        base.update({
            "ear_left":    ear + 0.01, "ear_right": ear - 0.01,
            "ear_avg":     ear, "perclos": perclos,
            "head_pitch":  lerp(20.0, 1.0, t),
            "head_state":  "forward" if t > 0.5 else "head_down",
            "is_distracted": False,
            "yawn_count":  2,
            "alert_level": 1 if t < 0.4 else 0,
            "alert_reason": "Recovering" if t < 0.6 else "Normal",
            "drowsiness_score": int(perclos * 6.67),
        })

    return base


def run_demo(speed: str = "normal"):
    mult = SPEEDS.get(speed, 1.0)

    phases = [
        (0, 30 * mult, "🟢 Phase 1: Normal driving"),
        (1, 30 * mult, "🟡 Phase 2: Early fatigue detected"),
        (2, 30 * mult, "🟠 Phase 3: Drowsy — warnings firing"),
        (3, 30 * mult, "🔴 Phase 4: CRITICAL — alarm triggered"),
        (4, 20 * mult, "🟢 Phase 5: Recovery after alarm"),
    ]

    print("\n" + "═"*55)
    print("  🎬 FocusGuard Demo Mode")
    print(f"  Speed: {speed}  |  Total: ~{int(sum(d for _,d,_ in phases))}s")
    print("  Dashboard: http://localhost:3000")
    print("═"*55 + "\n")

    for phase_idx, duration, label in phases:
        print(f"\n{label}")
        steps     = max(1, int(duration / 0.5))  # update every 500ms
        step_dur  = duration / steps

        for step in range(steps):
            t     = step / max(steps - 1, 1)
            state = build_state(phase_idx, t)
            try:
                r = requests.post(API_URL, json=state, timeout=2)
                if r.status_code != 200:
                    print(f"  ⚠ API error: {r.status_code}")
            except requests.exceptions.ConnectionError:
                print("  ✗ Cannot reach API at localhost:8000")
                print("    Make sure the server is running: python server/app.py")
                sys.exit(1)

            level  = state["alert_level"]
            icons  = ["✅","⚠️ ","🚨","🆘"]
            print(f"  {icons[level]} EAR={state['ear_avg']:.3f}  "
                  f"PERCLOS={state['perclos']:.1f}%  "
                  f"L{level}  head={state['head_state']:<12}", end="\r")
            time.sleep(step_dur)

        print()

    print("\n✅ Demo scenario complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FocusGuard Demo Mode")
    parser.add_argument("--speed", choices=["slow","normal","fast"],
                        default="normal",
                        help="Playback speed (default: normal)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop the demo continuously")
    args = parser.parse_args()

    try:
        while True:
            run_demo(args.speed)
            if not args.loop:
                break
            print("🔄 Looping demo in 3 seconds…")
            time.sleep(3)
    except KeyboardInterrupt:
        print("\n\nDemo stopped.")
