#!/usr/bin/env bash
# ==============================================================================
# O-RAN 3-Part Tariff Pricing — Fast Pipeline (경량 버전)
#
# [M10] 개발/디버깅용 경량 파이프라인  [Henderson 2018]
# 프로덕션 훈련은 scripts/run.sh 사용
#
# 변경 사항 vs run.sh:
#   - config/light.yaml 사용 (100K steps, 1 seed, 100 users)
#   - 평가 repeats: 5 → 2
#   - Streamlit 생략, 정적 PNG만 생성
#   - 별도 출력 디렉토리: outputs_light/
#
# Usage:
#   chmod +x scripts/run_fast.sh && ./scripts/run_fast.sh
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="config/light.yaml"
USERS_CSV="data/users_light.csv"
OUTPUT_DIR="outputs_light"

echo "============================================"
echo " O-RAN 3-Part Tariff — Fast Pipeline [M10]"
echo "============================================"
echo "Config:     $CONFIG"
echo "Output:     $OUTPUT_DIR"
echo "Project:    $PROJECT_DIR"
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

# ── Step 1: Generate user CSV (lightweight) ──
echo "===== Step 1: Generate Users (lightweight: 100) ====="
python -m oran3pt.gen_users \
    --config "$CONFIG" \
    --output "$USERS_CSV"
echo ""

# ── Step 2: Unit tests ──
echo "===== Step 2: Unit Tests ====="
python -m pytest tests/ -v --tb=short || echo "WARNING: Some tests failed (continuing)"
echo ""

# ── Step 3: Train SAC (lightweight: 100K steps, 1 seed) ──
echo "===== Step 3: Train SAC (Fast) ====="
python -m oran3pt.train \
    --config "$CONFIG" \
    --users "$USERS_CSV" \
    --output "$OUTPUT_DIR"
echo ""

# ── Step 4: Evaluate (lightweight: 2 repeats) ──
echo "===== Step 4: Evaluation (Fast) ====="
MODEL_PATH="$OUTPUT_DIR/best_model.zip"
if [ -f "$MODEL_PATH" ]; then
    echo "Found trained model: $MODEL_PATH"
    python -m oran3pt.eval \
        --config "$CONFIG" \
        --users "$USERS_CSV" \
        --model "$MODEL_PATH" \
        --output "$OUTPUT_DIR" \
        --repeats 2
else
    echo "No trained model found at $MODEL_PATH"
    echo "Evaluating with random policy (baseline only)"
    python -m oran3pt.eval \
        --config "$CONFIG" \
        --users "$USERS_CSV" \
        --output "$OUTPUT_DIR" \
        --repeats 2
fi
echo ""

# ── Step 5: Static dashboard only (skip Streamlit) ──
echo "===== Step 5: Dashboard (Static PNG) ====="
ROLLOUT_CSV="$OUTPUT_DIR/rollout_log.csv"
if [ -f "$ROLLOUT_CSV" ]; then
    python -c "
import pandas as pd
from pathlib import Path
from oran3pt.dashboard_app import _make_static_dashboard
df = pd.read_csv('$ROLLOUT_CSV')
_make_static_dashboard(df, Path('$OUTPUT_DIR'))
print('  Dashboard -> $OUTPUT_DIR/dashboard.png')
" || echo "  Dashboard generation skipped (matplotlib not available)"
fi
echo ""

# ── Step 5b: HTML convergence dashboard ──
echo "===== Step 5b: HTML Convergence Dashboard ====="
TRAIN_CSV="$OUTPUT_DIR/train_log_seed0.csv"
if [ -f "$TRAIN_CSV" ]; then
    echo "Generating HTML dashboard from: $TRAIN_CSV"
    python -m oran3pt.html_dashboard \
        --csv "$TRAIN_CSV" \
        --output "$OUTPUT_DIR/training_convergence_dashboard.html" \
        --seed 0 \
        --revision 8
    echo "  -> $OUTPUT_DIR/training_convergence_dashboard.html"
else
    echo "No training log found — skipping HTML dashboard"
fi
echo ""

# ── Step 5c: Comprehensive PNG dashboard sheets [M11] ──
echo "===== Step 5c: PNG Dashboard Sheets ====="
python -m oran3pt.png_dashboard \
    --output "$OUTPUT_DIR" \
    --config "$CONFIG" \
    --mode auto \
    --dpi 150 || echo "  PNG dashboard generation skipped"
echo ""

# ── Step 5d: 3D simulation dashboard [M12][M13] ──
echo "===== Step 5d: 3D Simulation Dashboard ====="
if [ -f "$ROLLOUT_CSV" ]; then
    SIM3D_ARGS="--csv $ROLLOUT_CSV --output $OUTPUT_DIR/sim3d_dashboard.html"
    [ -f "$USERS_CSV" ] && SIM3D_ARGS="$SIM3D_ARGS --users $USERS_CSV"
    EVENTS_CSV="$OUTPUT_DIR/user_events_log.csv"
    [ -f "$EVENTS_CSV" ] && SIM3D_ARGS="$SIM3D_ARGS --events $EVENTS_CSV"
    python -m oran3pt.sim3d_dashboard $SIM3D_ARGS || echo "  3D dashboard generation skipped"
    echo "  -> $OUTPUT_DIR/sim3d_dashboard.html"
else
    echo "No rollout log — skipping 3D dashboard"
fi
echo ""

echo "============================================"
echo " Fast Pipeline complete.  Outputs: $PROJECT_DIR/$OUTPUT_DIR/"
echo "============================================"
echo ""
echo " Files:"
ls -lh "$OUTPUT_DIR/" 2>/dev/null || echo "  (no outputs yet)"
