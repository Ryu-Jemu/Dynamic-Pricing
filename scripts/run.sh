#!/usr/bin/env bash
# ==============================================================================
# O-RAN 3-Part Tariff Pricing — End-to-End Pipeline  (Requirement 15)
#
# Steps:
#   0) Create venv & install dependencies
#   1) Generate synthetic user CSV
#   2) Run unit tests
#   3) Train SAC agent
#   4) Evaluate & export logs + CLV
#   5) Generate dashboard
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
pip install -r requirements.txt -q 2>&1 | tail -3
echo "Dependencies installed."

# Device info
python -c "
from oran3pt.utils import select_device
print(f'  PyTorch device: {select_device()}')
"
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
    python -m oran3pt.eval \
        --config config/default.yaml \
        --users data/users_init.csv \
        --model "$MODEL_PATH" \
        --repeats 5
else
    echo "No trained model found; evaluating with random policy"
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
