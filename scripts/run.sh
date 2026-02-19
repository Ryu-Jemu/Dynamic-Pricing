#!/usr/bin/env bash
# ==============================================================================
# O-RAN 3-Part Tariff Pricing — End-to-End Pipeline (v5.7)
#
# Mode selection (no interactive input):
#   ./scripts/run.sh light   — 경량 실행 (200K steps, 2 seeds, ~10분)
#   ./scripts/run.sh full    — 전체 실행 (1M steps, 5 seeds, ~2시간)
#
# Steps:
#   0) Clean previous outputs & caches
#   1) Create venv & install dependencies
#   2) Generate synthetic user CSV
#   3) Run unit tests
#   4) Train SAC agent (multi-seed with curriculum)
#   5) Evaluate & export logs + CLV + dashboards
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ── Mode selection ──
MODE="${1:-}"
if [ "$MODE" != "light" ] && [ "$MODE" != "full" ]; then
    echo ""
    echo "═══════════════════════════════════════════════"
    echo "  O-RAN 5G 3-Part Tariff SAC Training (v5.7)"
    echo "═══════════════════════════════════════════════"
    echo ""
    echo "  사용법: ./scripts/run.sh <모드>"
    echo ""
    echo "  모드 선택:"
    echo "    light  — 경량 실행 (200K steps, 2 seeds, ~10분)"
    echo "             개발/디버깅/빠른 검증용"
    echo ""
    echo "    full   — 전체 실행 (1M steps, 5 seeds, ~2시간)"
    echo "             Production 학습 + 통계적 유의성 확보"
    echo ""
    echo "  예시:"
    echo "    ./scripts/run.sh light"
    echo "    ./scripts/run.sh full"
    echo ""
    exit 1
fi

# ── Mode-specific configuration ──
if [ "$MODE" = "light" ]; then
    TIMESTEPS=100000
    N_SEEDS=1
    BUFFER_SIZE=50000
    EVAL_REPEATS=3
    OVERRIDE_FLAG=""
    MODE_LABEL="경량 (Light)"
    MODE_DESC="200K steps × 2 seeds"
else
    TIMESTEPS=""
    N_SEEDS=""
    BUFFER_SIZE=""
    EVAL_REPEATS=5
    OVERRIDE_FLAG="--override config/production.yaml"
    MODE_LABEL="전체 (Full / Production)"
    MODE_DESC="1M steps × 5 seeds"
fi

echo ""
echo "═══════════════════════════════════════════════"
echo "  O-RAN 5G 3-Part Tariff SAC Training (v5.7)"
echo "═══════════════════════════════════════════════"
echo "  Project : $PROJECT_DIR"
echo "  Mode    : $MODE_LABEL"
echo "  Config  : $MODE_DESC"
echo "═══════════════════════════════════════════════"
echo ""

# ── Step 0: Clean previous outputs & caches ──
echo "===== Step 0: Clean & Setup ====="

echo "  [0a] Cleaning previous outputs..."
if [ -d "outputs" ]; then
    PREV_FILES=$(find outputs -type f 2>/dev/null | wc -l | tr -d ' ')
    PREV_SIZE=$(du -sh outputs 2>/dev/null | cut -f1)
    rm -rf outputs
    echo "       Removed outputs/ ($PREV_FILES files, $PREV_SIZE)"
else
    echo "       outputs/ not found (clean state)"
fi
mkdir -p outputs

echo "  [0b] Cleaning Python caches..."
CACHE_COUNT=0
for d in $(find . -type d -name "__pycache__" -not -path "./.venv/*" -not -path "./.git/*" 2>/dev/null); do
    rm -rf "$d"
    CACHE_COUNT=$((CACHE_COUNT + 1))
done
rm -rf .pytest_cache .mypy_cache .ruff_cache
echo "       Removed $CACHE_COUNT __pycache__ dirs + tool caches"

echo "  [0c] Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "       Created new .venv"
else
    echo "       Using existing .venv"
fi
source .venv/bin/activate

echo "  [0d] Installing dependencies..."
pip install -q -r requirements.txt 2>&1 | tail -3
echo "       Dependencies installed."

# Verify critical packages
echo "  [0e] Verifying critical packages..."
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
        print(f'       {name} {ver}')
    except ImportError as e:
        errors.append(name)
        print(f'       {name} MISSING: {e}')
if errors:
    print(f'  FATAL: {len(errors)} missing packages: {errors}')
    sys.exit(1)
else:
    print('       All packages OK.')
" || { echo "ERROR: Critical packages missing. Aborting."; exit 1; }
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
echo "===== Step 3: Train SAC ($MODE_DESC) ====="
TRAIN_ARGS="--config config/default.yaml --users data/users_init.csv"
if [ -n "$OVERRIDE_FLAG" ]; then
    TRAIN_ARGS="$TRAIN_ARGS $OVERRIDE_FLAG"
fi
if [ -n "${TIMESTEPS:-}" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --timesteps $TIMESTEPS"
fi
if [ -n "${N_SEEDS:-}" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --seeds $N_SEEDS"
fi
if [ -n "${BUFFER_SIZE:-}" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --buffer-size $BUFFER_SIZE"
fi
python -m oran3pt.train $TRAIN_ARGS
echo ""

# ── Step 4: Evaluate ──
echo "===== Step 4: Evaluation (repeats=$EVAL_REPEATS) ====="
MODEL_PATH="outputs/best_model.zip"
if [ -f "$MODEL_PATH" ]; then
    echo "  Found trained model: $MODEL_PATH"
    python -m oran3pt.eval \
        --config config/default.yaml \
        --users data/users_init.csv \
        --model "$MODEL_PATH" \
        --repeats "$EVAL_REPEATS"
else
    echo "  No trained model found — evaluating random baseline"
    python -m oran3pt.eval \
        --config config/default.yaml \
        --users data/users_init.csv \
        --repeats "$EVAL_REPEATS"
fi
echo ""

# ── Summary ──
echo "═══════════════════════════════════════════════"
echo "  Pipeline complete!  Mode: $MODE_LABEL"
echo "  Outputs: $PROJECT_DIR/outputs/"
echo "═══════════════════════════════════════════════"
echo ""
echo "  Generated files:"
ls -lh outputs/ 2>/dev/null || echo "  (no outputs)"
echo ""
echo "  For interactive dashboard:"
echo "    streamlit run oran3pt/dashboard_app.py -- --data outputs/rollout_log.csv"
