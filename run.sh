#!/usr/bin/env bash
# ================================================================
# O-RAN 1-Cell Slicing + Pricing — End-to-End Pipeline
# Section 21: run.sh (must run end-to-end)
# ================================================================
#
# Usage:
#   chmod +x run.sh && ./run.sh
#
# This script:
#   1. Creates a Python virtual environment
#   2. Installs all dependencies
#   3. Runs calibration (demand + market + reward_scale)
#   4. Runs unit tests
#   5. Trains SAC agent
#   6. Evaluates trained agent (N repeats)
#   7. Generates plots + report.md
#   8. Prints device used and summary
# ================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="config/default.yaml"
CALIBRATED_CONFIG="config/calibrated.yaml"
VENV_DIR=".venv"
ARTIFACTS_DIR="artifacts"

# ---- Colors ----
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

step() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}========================================${NC}"
}

# ================================================================
# 1) Create venv & install dependencies
# ================================================================
step "1/7  Setting up virtual environment"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo "Created venv at $VENV_DIR"
else
    echo "Venv already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo -e "${GREEN}Dependencies installed.${NC}"

# ---- Print device ----
DEVICE=$(python3 -c "
from src.models.utils import select_device
d = select_device()
print(d)
")
echo -e "${YELLOW}Device: $DEVICE${NC}"

# ================================================================
# 2) Run calibration
# ================================================================
step "2/7  Calibration (demand + market + reward_scale)"

python3 -m src.models.calibrate --config "$CONFIG"

if [ -f "$CALIBRATED_CONFIG" ]; then
    echo -e "${GREEN}Calibrated config saved: $CALIBRATED_CONFIG${NC}"
    # Use calibrated config for subsequent steps
    RUN_CONFIG="$CALIBRATED_CONFIG"
else
    echo -e "${YELLOW}Warning: calibrated.yaml not found, using default config${NC}"
    RUN_CONFIG="$CONFIG"
fi

# ================================================================
# 3) Run unit tests
# ================================================================
step "3/7  Running unit tests"

python3 -m unittest discover -s tests -v
echo -e "${GREEN}All tests passed.${NC}"

# ================================================================
# 4) Train SAC
# ================================================================
step "4/7  Training SAC agent"

python3 -m src.train_sac --config "$RUN_CONFIG"

# Find the latest run directory
LATEST_RUN=$(ls -td "$ARTIFACTS_DIR"/*/ 2>/dev/null | head -1)
if [ -z "$LATEST_RUN" ]; then
    echo "Error: No run directory found in $ARTIFACTS_DIR"
    exit 1
fi
LATEST_RUN="${LATEST_RUN%/}"  # strip trailing slash
echo -e "${GREEN}Training complete. Run: $LATEST_RUN${NC}"

# ================================================================
# 5) Evaluate
# ================================================================
step "5/7  Evaluating trained agent"

python3 -m src.eval --config "$RUN_CONFIG" --run_dir "$LATEST_RUN"

echo -e "${GREEN}Evaluation complete.${NC}"

# ================================================================
# 6) Generate report
# ================================================================
step "6/7  Generating plots and report"

python3 -m src.report --run_dir "$LATEST_RUN"

echo -e "${GREEN}Report generated.${NC}"

# ================================================================
# 7) Summary
# ================================================================
step "7/7  Pipeline complete"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  O-RAN Slicing+Pricing Pipeline Complete   ${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "  Device:       $DEVICE"
echo "  Config:       $RUN_CONFIG"
echo "  Artifacts:    $LATEST_RUN"
echo "  Report:       $LATEST_RUN/report.md"
echo "  Plots:        $LATEST_RUN/plots/"
echo "  Eval CSV:     $LATEST_RUN/eval_monthly.csv"
echo "  Eval Summary: $LATEST_RUN/eval_summary.json"
echo "  Best Model:   $LATEST_RUN/best_model/"
echo "  Final Model:  $LATEST_RUN/final_model.zip"
echo ""

deactivate 2>/dev/null || true
