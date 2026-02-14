#!/usr/bin/env bash
# ==============================================================================
# O-RAN 3-Part Tariff Pricing — End-to-End Pipeline  (Requirement 15)
#
# REVISION 7 — Improvements from training evaluation:
#   [R1] 1M training timesteps (restored)
#   [R3] Curriculum learning (Phase 1: no churn/join; Phase 2: full dynamics)
#   [R8] Higher initial entropy coefficient
#   [NEW] HTML convergence dashboard generation (Step 5b)
#   Prior: [E9] Multi-seed training, [F1] transparent pip, [F2] SB3 verify
#
# Steps:
#   0) Create venv & install dependencies
#   1) Generate synthetic user CSV
#   2) Run unit tests
#   3) Train SAC agent (multi-seed with curriculum)
#   4) Evaluate & export logs + CLV
#   5a) Generate Streamlit / static PNG dashboard
#   5b) Generate HTML convergence dashboard
#
# Usage:
#   chmod +x scripts/run.sh && ./scripts/run.sh [--seeds N]
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Parse optional --seeds argument
N_SEEDS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds) N_SEEDS="$2"; shift 2;;
        *) shift;;
    esac
done

echo "============================================"
echo " O-RAN 3-Part Tariff — Full Pipeline (v7)"
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

echo "Installing dependencies..."
pip install -r requirements.txt 2>&1 | tee /tmp/pip_install.log | tail -5
echo ""

# Verify critical packages
echo "Verifying critical packages..."
python -c "
import sys
errors = []

try:
    import stable_baselines3
    print(f'  stable-baselines3 {stable_baselines3.__version__}')
except ImportError as e:
    errors.append(f'stable-baselines3: {e}')
    print(f'  stable-baselines3 MISSING: {e}')

try:
    import torch
    from oran3pt.utils import select_device
    dev = select_device()
    print(f'  torch {torch.__version__} (device: {dev})')
except ImportError as e:
    errors.append(f'torch: {e}')
    print(f'  torch MISSING: {e}')

try:
    import gymnasium
    print(f'  gymnasium {gymnasium.__version__}')
except ImportError as e:
    errors.append(f'gymnasium: {e}')
    print(f'  gymnasium MISSING: {e}')

if errors:
    print()
    print('  Some packages failed. Attempting targeted install...')
    sys.exit(1)
else:
    print('  All critical packages verified.')
" || {
    echo ""
    echo "Attempting targeted SB3 install..."
    pip install "stable-baselines3[extra]>=2.3.0" --force-reinstall 2>&1 | tail -5
    python -c "import stable_baselines3; print(f'  SB3 {stable_baselines3.__version__} installed')" || {
        echo "  SB3 install failed. Training will use random baseline."
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
TRAIN_ARGS="--config config/default.yaml --users data/users_init.csv"
if [ -n "$N_SEEDS" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --seeds $N_SEEDS"
fi
python -m oran3pt.train $TRAIN_ARGS
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
    echo "No trained model found at $MODEL_PATH"
    echo "Evaluating with random policy (baseline only)"
    python -m oran3pt.eval \
        --config config/default.yaml \
        --users data/users_init.csv \
        --repeats 5
fi
echo ""

# ── Step 5a: Streamlit / static dashboard ──
echo "===== Step 5a: Dashboard (Streamlit / PNG) ====="
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

# ── Step 5b: HTML convergence dashboard ──
echo "===== Step 5b: HTML Convergence Dashboard ====="
# Find the training log for seed 0
TRAIN_CSV="outputs/train_log_seed0.csv"
if [ -f "$TRAIN_CSV" ]; then
    echo "Generating HTML dashboard from: $TRAIN_CSV"
    python -m oran3pt.html_dashboard \
        --csv "$TRAIN_CSV" \
        --output outputs/training_convergence_dashboard.html \
        --seed 0 \
        --revision 7
    echo "  → outputs/training_convergence_dashboard.html"
else
    echo "No training log found at $TRAIN_CSV"
    # Fall back to rollout log if available
    if [ -f "outputs/rollout_log.csv" ]; then
        echo "Using evaluation rollout log instead"
        python -m oran3pt.html_dashboard \
            --csv outputs/rollout_log.csv \
            --output outputs/training_convergence_dashboard.html \
            --seed 0 \
            --revision 7
        echo "  → outputs/training_convergence_dashboard.html"
    else
        echo "No CSV logs found — skipping HTML dashboard"
    fi
fi
echo ""

# ── Step 5c: Comprehensive PNG dashboard sheets ──
echo "===== Step 5c: PNG Dashboard Sheets [M11] ====="
python -m oran3pt.png_dashboard \
    --output outputs \
    --config config/default.yaml \
    --mode auto \
    --dpi 180
echo ""

# ── Step 5d: 3D simulation dashboard ──
echo "===== Step 5d: 3D Simulation Dashboard [M12][M13] ====="
ROLLOUT_CSV="outputs/rollout_log.csv"
USERS_CSV="data/users_init.csv"
EVENTS_CSV="outputs/user_events_log.csv"
if [ -f "$ROLLOUT_CSV" ]; then
    SIM3D_ARGS="--csv $ROLLOUT_CSV --output outputs/sim3d_dashboard.html"
    [ -f "$USERS_CSV" ] && SIM3D_ARGS="$SIM3D_ARGS --users $USERS_CSV"
    [ -f "$EVENTS_CSV" ] && SIM3D_ARGS="$SIM3D_ARGS --events $EVENTS_CSV"
    python -m oran3pt.sim3d_dashboard $SIM3D_ARGS
    echo "  → outputs/sim3d_dashboard.html"
else
    echo "No rollout log found — skipping 3D dashboard"
fi
echo ""

echo "============================================"
echo " Pipeline complete.  Outputs: $PROJECT_DIR/outputs/"
echo "============================================"
echo ""
echo " Files:"
ls -lh outputs/ 2>/dev/null || echo "  (no outputs yet)"
