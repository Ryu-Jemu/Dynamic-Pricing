#!/usr/bin/env bash
# ==============================================================================
# O-RAN 1-Cell Slicing + Pricing — End-to-End Pipeline (§21)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "===== O-RAN Slicing Pipeline ====="

# ── Step 0: Device info ──
python -c "
from src.models.utils import select_device
dev = select_device()
print(f'Device: {dev}')
"

# ── Step 1: Run tests ──
echo ""
echo "===== Step 1: Tests ====="
python -m pytest tests/ -v --tb=short || echo "WARNING: Some tests failed"

# ── Step 2: Calibrate ──
echo ""
echo "===== Step 2: Calibration ====="
python -m src.models.calibrate --config config/default.yaml --output config/calibrated.yaml

# ── Step 3: Train ──
echo ""
echo "===== Step 3: Training ====="
python -m src.train_sac --config config/calibrated.yaml

# ── Step 4: Evaluate ──
echo ""
echo "===== Step 4: Evaluation ====="
RUN_DIR=$(ls -td artifacts/*/ 2>/dev/null | head -1)
if [ -z "$RUN_DIR" ]; then
    RUN_DIR="artifacts/latest"
    mkdir -p "$RUN_DIR"
fi
echo "Using run dir: $RUN_DIR"

MODEL_PATH="${RUN_DIR}best_model.zip"
if [ -f "$MODEL_PATH" ]; then
    python -m src.eval --config config/calibrated.yaml \
                       --model "$MODEL_PATH" \
                       --output "${RUN_DIR}eval.csv" \
                       --repeats 3
else
    echo "No trained model found; evaluating with random policy"
    python -m src.eval --config config/calibrated.yaml \
                       --output "${RUN_DIR}eval.csv" \
                       --repeats 3
fi

# ── Step 5: Report ──
echo ""
echo "===== Step 5: Report ====="
python -m src.report --run_dir "$RUN_DIR"

echo ""
echo "===== Pipeline complete ====="
