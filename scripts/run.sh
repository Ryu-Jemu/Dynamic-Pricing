#!/usr/bin/env bash
# ==============================================================================
# O-RAN 3-Part Tariff Pricing — End-to-End Pipeline  (Requirement 15)
#
# v10.4 pipeline:
#   [R1] 1M training timesteps (restored)
#   [R3] Curriculum learning (3-phase)
#   [R8] Higher initial entropy coefficient
#   [M15] Dashboard generation integrated into eval.py (Step 4)
#   Prior: [E9] Multi-seed training, [F1] transparent pip, [F2] SB3 verify
#
# Steps:
#   0) Create venv & install dependencies
#   1) Generate synthetic user CSV
#   2) Run unit tests
#   3) Train SAC agent (multi-seed with curriculum)
#   4) Evaluate & export logs + CLV + dashboards [M15]
#
# Usage:
#   chmod +x scripts/run.sh && ./scripts/run.sh [--seeds N] [--auto]
#   --auto : Skip interactive prompts, use config defaults
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Parse CLI arguments
N_SEEDS=""
TIMESTEPS=""
BUFFER_SIZE=""
BATCH_SIZE=""
LR=""
AUTO_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds) N_SEEDS="$2"; shift 2;;
        --timesteps) TIMESTEPS="$2"; shift 2;;
        --buffer-size) BUFFER_SIZE="$2"; shift 2;;
        --batch-size) BATCH_SIZE="$2"; shift 2;;
        --lr) LR="$2"; shift 2;;
        --auto) AUTO_MODE=true; shift;;
        *) shift;;
    esac
done

echo "═══════════════════════════════════════════════"
echo "  O-RAN 5G 3-Part Tariff SAC Training (v10.4)"
echo "═══════════════════════════════════════════════"
echo "Project dir: $PROJECT_DIR"
echo ""

# ── Interactive hyperparameter input ──
# Skipped if --auto flag is provided or all values are set via CLI
if [ "$AUTO_MODE" = false ]; then
    echo "  하이퍼파라미터 설정 (Enter = 기본값 사용)"
    echo "  ──────────────────────────────────────────"

    if [ -z "$TIMESTEPS" ]; then
        read -p "  Total timesteps [1000000 권장, 개발: 100000]: " TS_INPUT
        TIMESTEPS="${TS_INPUT:-}"
    fi

    if [ -z "$N_SEEDS" ]; then
        read -p "  Number of seeds [5 권장, 개발: 1]: " SEEDS_INPUT
        N_SEEDS="${SEEDS_INPUT:-}"
    fi

    if [ -z "$BUFFER_SIZE" ]; then
        read -p "  Replay buffer size [200000 권장, 개발: 50000]: " BUF_INPUT
        BUFFER_SIZE="${BUF_INPUT:-}"
    fi

    if [ -z "$BATCH_SIZE" ]; then
        read -p "  Batch size [512 권장]: " BS_INPUT
        BATCH_SIZE="${BS_INPUT:-}"
    fi

    if [ -z "$LR" ]; then
        read -p "  Learning rate [0.0003 권장]: " LR_INPUT
        LR="${LR_INPUT:-}"
    fi

    echo ""
    echo "  설정 요약:"
    echo "    Timesteps:     ${TIMESTEPS:-config default}"
    echo "    Seeds:         ${N_SEEDS:-config default}"
    echo "    Buffer:        ${BUFFER_SIZE:-config default}"
    echo "    Batch size:    ${BATCH_SIZE:-config default}"
    echo "    Learning rate: ${LR:-config default}"
    echo ""
    read -p "  계속하시겠습니까? [Y/n]: " CONFIRM
    CONFIRM="${CONFIRM:-Y}"
    if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
        echo "  중단되었습니다."
        exit 0
    fi
    echo ""
fi

# ── Step 0: Clean environment ──
echo "===== Step 0: Environment Setup ====="
echo "Cleaning previous caches..."
find . -type d -name "__pycache__" -not -path "./.git/*" -exec rm -rf {} + 2>/dev/null || true
rm -rf .pytest_cache
echo "  Removed __pycache__ and .pytest_cache"

echo "Recreating virtual environment..."
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate

echo "Clearing pip cache (prevents AssertionError from corrupted wheels)..."
pip cache purge 2>/dev/null || true

echo "Installing dependencies..."
pip install -r requirements.txt 2>&1 | tee /tmp/pip_install.log
PIP_EXIT=${PIPESTATUS[0]}
if [ "$PIP_EXIT" -ne 0 ]; then
    echo ""
    echo "ERROR: pip install failed (exit code $PIP_EXIT). Last 10 lines:"
    tail -10 /tmp/pip_install.log
    echo ""
    echo "Attempting fresh install without cache..."
    pip install --no-cache-dir -r requirements.txt 2>&1 | tee /tmp/pip_install.log
    PIP_EXIT=${PIPESTATUS[0]}
    if [ "$PIP_EXIT" -ne 0 ]; then
        echo "ERROR: pip install failed again. Aborting."
        exit 1
    fi
fi
echo ""

# Verify ALL critical packages (not just SB3)
echo "Verifying critical packages..."
python -c "
import sys
errors = []
for mod, name in [
    ('numpy', 'numpy'), ('scipy', 'scipy'), ('pandas', 'pandas'),
    ('torch', 'torch'), ('gymnasium', 'gymnasium'),
    ('stable_baselines3', 'stable-baselines3'),
    ('matplotlib', 'matplotlib'), ('yaml', 'pyyaml'),
]:
    try:
        m = __import__(mod)
        ver = getattr(m, '__version__', 'OK')
        print(f'  {name} {ver}')
    except ImportError as e:
        errors.append(name)
        print(f'  {name} MISSING: {e}')

if errors:
    print()
    print(f'  FATAL: {len(errors)} missing packages: {errors}')
    sys.exit(1)
else:
    print('  All critical packages verified.')
" || {
    echo ""
    echo "ERROR: Critical packages missing after install. Aborting."
    echo "Check /tmp/pip_install.log for details."
    exit 1
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
if [ -n "$TIMESTEPS" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --timesteps $TIMESTEPS"
fi
if [ -n "$BUFFER_SIZE" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --buffer-size $BUFFER_SIZE"
fi
if [ -n "$BATCH_SIZE" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --batch-size $BATCH_SIZE"
fi
if [ -n "$LR" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --lr $LR"
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

# [M15] Dashboards are now auto-generated by eval.py (Step 4).
# For interactive Streamlit dashboard, run separately:
#   streamlit run oran3pt/dashboard_app.py -- --data outputs/rollout_log.csv

echo "============================================"
echo " Pipeline complete.  Outputs: $PROJECT_DIR/outputs/"
echo "============================================"
echo ""
echo " Files:"
ls -lh outputs/ 2>/dev/null || echo "  (no outputs yet)"
