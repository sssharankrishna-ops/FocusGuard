#!/bin/bash
# setup.sh — FocusGuard one-command bootstrap
# Usage: bash scripts/setup.sh

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo ""
echo "══════════════════════════════════════════"
echo "  🚗 FocusGuard Setup"
echo "══════════════════════════════════════════"
echo ""

# ── Python venv ───────────────────────────────
echo "→ Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "→ Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  ✓ Python deps installed"

# ── Directories ───────────────────────────────
mkdir -p data sounds reports models/mobilenet_ssd
echo "  ✓ Directories created"

# ── Generate sound files ───────────────────────
echo "→ Generating alert sound files..."
python3 - <<'PYEOF'
import sys
sys.path.insert(0, '.')
from vision.alert_engine import ensure_sounds
ensure_sounds()
print("  ✓ Sound files ready")
PYEOF

# ── MobileNet SSD model download ──────────────
PROTO="models/mobilenet_ssd/deploy.prototxt"
MODEL="models/mobilenet_ssd/mobilenet_iter_73000.caffemodel"

if [ ! -f "$PROTO" ] || [ ! -f "$MODEL" ]; then
  echo "→ Downloading MobileNet SSD model files (~24MB)..."
  curl -L -o "$PROTO" \
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt" \
    2>/dev/null || echo "  ⚠ Could not download prototxt (phone detection will be disabled)"
  curl -L -o "$MODEL" \
    "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc" \
    2>/dev/null || echo "  ⚠ Could not download caffemodel (phone detection will be disabled)"
fi

# ── Init DB ────────────────────────────────────
echo "→ Initialising database..."
python3 -c "
import sys; sys.path.insert(0,'.')
from server.database import IncidentLogger
IncidentLogger()
print('  ✓ Database ready at data/focusguard.db')
"

# ── React dashboard ────────────────────────────
echo "→ Setting up React dashboard..."
if [ ! -d "dashboard/node_modules" ]; then
  cd dashboard
  npm install --silent
  cd ..
  echo "  ✓ Node dependencies installed"
else
  echo "  ✓ Node dependencies already present"
fi

# ── .env ──────────────────────────────────────
if [ ! -f ".env" ]; then
  cat > .env <<'ENVEOF'
CAMERA_INDEX=0
WEBSOCKET_PORT=8000
SENSITIVITY=medium
ENVEOF
  echo "  ✓ .env created"
fi

echo ""
echo "══════════════════════════════════════════"
echo "  ✅ Setup complete!"
echo ""
echo "  To run FocusGuard:"
echo ""
echo "  Terminal 1 — Backend:"
echo "    source venv/bin/activate"
echo "    python server/app.py"
echo ""
echo "  Terminal 2 — Vision engine:"
echo "    source venv/bin/activate"
echo "    python vision/main.py"
echo ""
echo "  Terminal 3 — React dashboard:"
echo "    cd dashboard && npm start"
echo ""
echo "  Demo mode (no camera needed):"
echo "    python scripts/demo_mode.py"
echo ""
echo "  Dashboard URL: http://localhost:3000"
echo "══════════════════════════════════════════"
echo ""
