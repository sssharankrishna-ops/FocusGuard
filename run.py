"""
run.py — FocusGuard launcher.
Starts the FastAPI server + vision bridge in one command.

Usage:
    python run.py                   # real camera
    python run.py --demo            # demo mode (no camera)
    python run.py --sensitivity high
    python run.py --camera 1        # use camera index 1
"""

import subprocess
import threading
import time
import sys
import os
import argparse
import signal
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("focusguard.run")

ROOT    = os.path.dirname(os.path.abspath(__file__))
PYTHON  = sys.executable
processes = []


def start_proc(name: str, cmd: list, cwd: str = ROOT) -> subprocess.Popen:
    logger.info(f"Starting {name}...")
    p = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    processes.append((name, p))
    return p


def stream_output(name: str, proc: subprocess.Popen):
    """Print process stdout with prefix."""
    prefix_colors = {
        "server":  "\033[34m",  # blue
        "vision":  "\033[32m",  # green
        "demo":    "\033[33m",  # yellow
    }
    color = prefix_colors.get(name, "\033[0m")
    reset = "\033[0m"
    for line in proc.stdout:
        print(f"{color}[{name}]{reset} {line}", end="")


def shutdown(sig=None, frame=None):
    print("\n\n🛑  Shutting down FocusGuard...\n")
    for name, p in processes:
        try:
            p.terminate()
            logger.info(f"Stopped {name}")
        except Exception:
            pass
    sys.exit(0)


def wait_for_server(timeout: int = 15) -> bool:
    """Poll until FastAPI is ready."""
    import urllib.request
    for i in range(timeout * 2):
        try:
            urllib.request.urlopen("http://localhost:8000/", timeout=1)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser(description="FocusGuard launcher")
    parser.add_argument("--demo",        action="store_true",
                        help="Run in demo mode (no camera required)")
    parser.add_argument("--camera",      type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--sensitivity", type=str, default="medium",
                        choices=["low", "medium", "high"],
                        help="Alert sensitivity (default: medium)")
    parser.add_argument("--no-display",  action="store_true",
                        help="Run vision engine headless (no OpenCV window)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("\n" + "═" * 52)
    print("  🚗  FocusGuard — Starting up")
    print("═" * 52)

    # ── 1. FastAPI server ──────────────────────────────────────────
    server_proc = start_proc(
        "server",
        [PYTHON, "-m", "uvicorn", "server.app:app",
         "--host", "0.0.0.0", "--port", "8000", "--log-level", "warning"],
        cwd=ROOT
    )
    threading.Thread(
        target=stream_output, args=("server", server_proc), daemon=True
    ).start()

    # Wait for server to be ready before starting vision
    logger.info("Waiting for API server to be ready...")
    if not wait_for_server(timeout=15):
        logger.error("Server did not start in time. Check server/app.py for errors.")
        shutdown()

    logger.info("✓ API server ready at http://localhost:8000")
    print("\n  Dashboard:  http://localhost:3000")
    print("  API docs:   http://localhost:8000/docs")
    print("  Press Ctrl+C to stop all services.\n")

    # ── 2. Vision engine or demo ───────────────────────────────────
    if args.demo:
        time.sleep(0.5)
        demo_proc = start_proc(
            "demo",
            [PYTHON, "scripts/demo_mode.py", "--loop"],
            cwd=ROOT
        )
        threading.Thread(
            target=stream_output, args=("demo", demo_proc), daemon=True
        ).start()
        logger.info("✓ Demo mode running (looping scenario)")
    else:
        vision_cmd = [
            PYTHON, "vision/bridge.py",
            "--camera",      str(args.camera),
            "--sensitivity", args.sensitivity,
        ]
        if args.no_display:
            vision_cmd.append("--no-display")

        # bridge.py starts vision internally; use direct main for simplicity
        vision_cmd = [
            PYTHON, "-c",
            f"""
import sys, os
sys.path.insert(0, r'{ROOT}')
from vision.main import run
run(camera_idx={args.camera},
    sensitivity='{args.sensitivity}',
    no_display={args.no_display})
"""
        ]
        vision_proc = start_proc("vision", vision_cmd, cwd=ROOT)
        threading.Thread(
            target=stream_output, args=("vision", vision_proc), daemon=True
        ).start()
        logger.info(f"✓ Vision engine running (camera {args.camera})")

    # ── 3. Keep alive ──────────────────────────────────────────────
    try:
        while True:
            # Check if critical processes are still alive
            for name, p in processes:
                if p.poll() is not None:
                    logger.error(f"Process '{name}' exited unexpectedly (code {p.returncode}). Shutting down.")
                    shutdown()
            time.sleep(2)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
