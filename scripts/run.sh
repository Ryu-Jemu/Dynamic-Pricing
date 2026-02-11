#!/usr/bin/env bash
# ==============================================================================
# O-RAN 3-Part Tariff Pricing — End-to-End Pipeline  (Requirement 15)
#
# Steps:
#   0) Create venv & install dependencies (with SB3 verification)
#   1) Generate synthetic user CSV
#   2) Run unit tests
#   3) Train SAC agent
#   4) Evaluate & export logs + CLV
#   5) Generate dashboard
#
# REVISION 3 — Fixes:
#   [F1] pip install no longer suppresses errors (was hiding SB3 failure)
#   [F2] Explicit SB3 import check before training
#   [F3] Fallback: attempt SB3 install from PyPI if missing
#
# Usage:
#   chmod +x scripts/run.sh && ./scripts/run.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================"
echo " O-RAN 3-Part Tariff — Full Pipeline"
echo "============================================"
echo "Project dir: $PROJECT_DIR"
echo ""

# ── Step 0: Virtual environment ──
echo "===== Step 0: Environment Setup ====="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created virtual environment"
fi
source .venv/bin/activate
pip install --upgrade pip -q

# [F1] FIX: Do NOT suppress pip output — show errors clearly
echo "Installing dependencies..."
pip install -r requirements.txt 2>&1 | tee /tmp/pip_install.log | tail -5
echo ""

# [F2] FIX: Verify SB3 is importable before proceeding
echo "Verifying critical packages..."
python -c "
import sys
errors = []

# Check SB3
try:
    import stable_baselines3
    print(f'  ✅ stable-baselines3 {stable_baselines3.__version__}')
except ImportError as e:
    errors.append(f'stable-baselines3: {e}')
    print(f'  ❌ stable-baselines3 MISSING: {e}')

# Check PyTorch
try:
    import torch
    from oran3pt.utils import select_device
    dev = select_device()
    print(f'  ✅ torch {torch.__version__} (device: {dev})')
except ImportError as e:
    errors.append(f'torch: {e}')
    print(f'  ❌ torch MISSING: {e}')

# Check gymnasium
try:
    import gymnasium
    print(f'  ✅ gymnasium {gymnasium.__version__}')
except ImportError as e:
    errors.append(f'gymnasium: {e}')
    print(f'  ❌ gymnasium MISSING: {e}')

if errors:
    print()
    print('  ⚠️  Some packages failed to install.')
    print('  Attempting targeted install of missing packages...')
    sys.exit(1)
else:
    print('  All critical packages verified.')
" || {
    # [F3] FIX: Attempt targeted reinstall of SB3
    echo ""
    echo "Attempting targeted SB3 install..."
    pip install "stable-baselines3[extra]>=2.3.0" --force-reinstall 2>&1 | tail -5
    python -c "import stable_baselines3; print(f'  ✅ SB3 {stable_baselines3.__version__} installed successfully')" || {
        echo "  ❌ SB3 install failed. Training will use random baseline."
        echo "  Common causes: incompatible torch version, missing C++ compiler"
        echo "  Try: pip install stable-baselines3 --verbose"
    }
}
echo ""

# ── Step 1: Generate user CSV ──
echo "===== Step 1: Generate Users ====="
python -m oran3pt.gen_users \
    --config config/default.yaml \
    --output data/users_init.csv
echo ""

# ── Step 2: Unit tests ──
echo "===== Step 2: Unit Tests ====="
python -m pytest tests/ -v --tb=short || echo "WARNING: Some tests failed (continuing)"
echo ""

# ── Step 3: Train SAC ──
echo "===== Step 3: Train SAC ====="
python -m oran3pt.train \
    --config config/default.yaml \
    --users data/users_init.csv
echo ""

# ── Step 4: Evaluate ──
echo "===== Step 4: Evaluation ====="
MODEL_PATH="outputs/best_model.zip"
if [ -f "$MODEL_PATH" ]; then
    echo "Found trained model: $MODEL_PATH"
    python -m oran3pt.eval \
        --config config/default.yaml \
        --users data/users_init.csv \
        --model "$MODEL_PATH" \
        --repeats 5
else
    echo "⚠️  No trained model found at $MODEL_PATH"
    echo "   Evaluating with random policy (results will be baseline only)"
    python -m oran3pt.eval \
        --config config/default.yaml \
        --users data/users_init.csv \
        --repeats 5
fi
echo ""

# ── Step 5: Dashboard ──
echo "===== Step 5: Dashboard ====="
if python -c "import streamlit" 2>/dev/null; then
    echo "Launching Streamlit dashboard..."
    echo "  URL: http://localhost:8501"
    streamlit run oran3pt/dashboard_app.py -- --data outputs/rollout_log.csv
else
    echo "Streamlit not available — generating static PNG dashboard"
    python -c "
import pandas as pd
from pathlib import Path
from oran3pt.dashboard_app import _make_static_dashboard
df = pd.read_csv('outputs/rollout_log.csv')
_make_static_dashboard(df, Path('outputs'))
"
fi

echo ""
echo "============================================"
echo " Pipeline complete."
echo " Outputs in: $PROJECT_DIR/outputs/"
echo "============================================"