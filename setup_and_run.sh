#!/bin/bash
#═══════════════════════════════════════════════════════════════════
#  5G O-RAN Network Slicing - 맥북 설치 및 실행 스크립트
#═══════════════════════════════════════════════════════════════════
#
#  한 번에 실행:
#    chmod +x setup_and_run.sh && ./setup_and_run.sh
#
#  옵션:
#    ./setup_and_run.sh --fast    빠른 테스트 (10K steps)
#    ./setup_and_run.sh --full    전체 학습 (500K steps)
#
#═══════════════════════════════════════════════════════════════════

set -e

# 색상
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 기본 설정
TIMESTEPS=100000
FAST=""

# 인자 파싱
for arg in "$@"; do
    case $arg in
        --fast) TIMESTEPS=10000; FAST="--fast" ;;
        --full) TIMESTEPS=500000 ;;
        -h|--help)
            echo "사용법: ./setup_and_run.sh [--fast|--full]"
            echo "  --fast  빠른 테스트 (10K steps)"
            echo "  --full  전체 학습 (500K steps)"
            exit 0 ;;
    esac
done

cd "$(dirname "$0")"

echo -e "${BLUE}"
echo "═══════════════════════════════════════════════════════════════"
echo "  🚀 5G O-RAN RL Simulation - Setup & Run"
echo "═══════════════════════════════════════════════════════════════"
echo -e "${NC}"

# Python 확인
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "❌ Python not found"
    exit 1
fi
echo -e "${GREEN}✓ Python: $($PYTHON --version)${NC}"

# 의존성 설치
echo -e "${YELLOW}📦 Installing dependencies...${NC}"
$PYTHON -m pip install --upgrade pip -q
$PYTHON -m pip install -r requirements.txt -q
echo -e "${GREEN}✓ Dependencies installed${NC}"

# MPS 확인
MPS=$($PYTHON -c "import torch; print('✓' if torch.backends.mps.is_available() else '✗')" 2>/dev/null || echo "✗")
if [ "$MPS" = "✓" ]; then
    echo -e "${GREEN}✓ Apple MPS: Available${NC}"
else
    echo -e "${YELLOW}⚠ Apple MPS: Not available (CPU mode)${NC}"
fi

# 학습 실행
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}Starting training: ${TIMESTEPS} steps${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

$PYTHON run_training.py --timesteps $TIMESTEPS $FAST

echo ""
echo -e "${GREEN}✅ Complete!${NC}"
