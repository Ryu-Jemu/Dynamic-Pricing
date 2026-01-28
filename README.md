# 5G O-RAN Network Slicing RL Simulation

**Dynamic Pricing Optimization using Constrained MDP and SAC**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements a research-ready simulation framework for **5G O-RAN network slicing** with **Three-Part Tariff pricing optimization** using **Constrained Markov Decision Process (CMDP)** and **Soft Actor-Critic (SAC)** reinforcement learning.

### Key Features

- **Standards-Compliant**: 3GPP TR 38.901 channel model, NR PRB tables from 3GPP 38.101
- **Three-Part Tariff**: Access fee (F) + Allowance (D) + Overage price (p) per slice
- **CMDP Framework**: Primal-dual Lagrangian optimization with constraint satisfaction
- **Online User Dynamics**: NHPP/Hawkes arrivals + hazard-based churn with overage burden
- **Realistic Cost Model**: Energy, spectrum, backhaul decomposition
- **Calibration Pipeline**: MLE/Bayesian parameter estimation with confidence intervals

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Near-RT RIC (O-RAN)                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Pricing xApp (CMDP-SAC Agent)                │  │
│  │  • Observation: Users, QoS, Prices, Allowance state       │  │
│  │  • Action: [F_urllc, p_urllc, F_embb, p_embb]            │  │
│  │  • Constraints: URLLC/eMBB violation rates                │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ E2 Interface
┌────────────────────────────▼────────────────────────────────────┐
│                        gNB (5G Base Station)                    │
│  ┌────────────────────┐  ┌────────────────────┐                │
│  │   URLLC Slice      │  │    eMBB Slice      │                │
│  │   (B2B, 99.999%)   │  │    (B2C, 99.9%)    │                │
│  └────────────────────┘  └────────────────────┘                │
│  • 20 MHz @ 30 kHz SCS → 51 PRBs (3GPP 38.101)                 │
│  • Two-stage scheduling: Slice-level + In-slice PF/Priority    │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

### 맥북 (Apple Silicon - 권장)

```bash
# 한 번에 설치 및 실행
chmod +x setup_and_run.sh
./setup_and_run.sh --fast   # 빠른 테스트 (10K 스텝)

# 또는 전체 학습
./setup_and_run.sh --full   # 전체 학습 (500K 스텝)
```

### 수동 설치

```bash
# Clone repository
git clone <repository-url>
cd 5G_ORAN_RL_Simulation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start (한 번에 실행)

```bash
# 방법 1: 스크립트 사용 (권장)
./setup_and_run.sh --fast

# 방법 2: Python 직접 실행
python run_training.py --timesteps 100000

# 방법 3: 빠른 테스트 (계산량 최적화)
python run_training.py --timesteps 10000 --fast

# 방법 4: 전체 학습
python run_training.py --timesteps 500000
```

### 명령어 옵션

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--timesteps`, `-t` | 학습 스텝 수 | `-t 100000` |
| `--fast`, `-f` | 빠른 모드 (계산량 감소) | `--fast` |
| `--device`, `-d` | 디바이스 선택 | `-d mps` |
| `--seed`, `-s` | 랜덤 시드 | `-s 42` |

### 하드웨어 가속

| 환경 | 가속 | 설정 |
|------|------|------|
| MacBook (M1/M2/M3) | MPS | 자동 감지 |
| NVIDIA GPU | CUDA | 자동 감지 |
| CPU | 없음 | `--device cpu` |

### Requirements

- Python 3.9+
- numpy >= 1.24.0
- scipy >= 1.10.0
- gymnasium >= 0.29.0
- stable-baselines3 >= 2.2.0
- torch >= 2.0.0 (MPS 지원)
- tqdm >= 4.65.0 (진행률 표시)
- pandas >= 2.0.0
- matplotlib >= 3.7.0

## Project Structure

```
5G_ORAN_RL_Simulation/
├── config/
│   └── scenario_config.py      # All parameters (no hardcoded constants)
├── env/
│   ├── nr_prb_table.py         # 3GPP 38.101 PRB derivation
│   ├── channel_38901.py        # 3GPP TR 38.901 UMi channel
│   ├── qos_fbl.py              # URLLC FBL reliability
│   ├── qos_embb.py             # eMBB throughput/MCS
│   ├── scheduler.py            # Two-stage PRB allocation
│   └── network_slicing_cmdp_env.py  # Gymnasium CMDP environment
├── models/
│   ├── tariff_three_part.py    # Three-Part Tariff billing
│   ├── arrivals_nhpp.py        # NHPP user arrivals
│   ├── arrivals_hawkes.py      # Hawkes process (optional)
│   ├── churn_hazard.py         # Hazard-based churn
│   └── costs.py                # Cost decomposition
├── training/
│   └── train_cmdp_sac.py       # Primal-dual Lagrangian SAC
├── calibration/
│   ├── fit_demand.py           # Arrival parameter estimation
│   ├── fit_churn.py            # Churn coefficient estimation
│   └── fit_energy.py           # Energy model calibration
├── evaluation/
│   └── plots.py                # Visualization and analysis
├── utils/
│   ├── logger.py               # Logging and metrics tracking
│   └── helpers.py              # Common utilities
├── logs/                       # Training logs
├── results/                    # Evaluation outputs
├── requirements.txt
└── README.md
```

## Usage

### 1. Configuration

Edit `config/scenario_config.py` to customize:

```python
# Bandwidth and PRB configuration
BANDWIDTH_MHZ = 20
SCS_KHZ = 30  # → 51 PRBs derived from 3GPP table

# Service parameters
URLLC_LATENCY_DEADLINE_MS = 1.0
URLLC_TARGET_BLER = 1e-5
EMBB_TARGET_THROUGHPUT_MBPS = 50.0

# Pricing bounds
URLLC_FEE_BOUNDS = (40.0, 60.0)  # $/hour
EMBB_FEE_BOUNDS = (3.0, 8.0)     # $/hour

# Constraint thresholds
URLLC_VIOLATION_THRESHOLD = 0.001
EMBB_VIOLATION_THRESHOLD = 0.01
```

### 2. Training

```bash
# Train CMDP-SAC agent
python -m training.train_cmdp_sac \
    --total_timesteps 750000 \
    --seed 42 \
    --experiment_name my_experiment

# With custom configuration
python -m training.train_cmdp_sac \
    --learning_rate 2e-4 \
    --batch_size 512 \
    --constraint_threshold_urllc 0.001 \
    --constraint_threshold_embb 0.01
```

### 3. Calibration (Optional)

```bash
# Fit demand model from data
python -m calibration.fit_demand \
    --data_path data/arrivals.csv \
    --service_type eMBB

# Fit churn model
python -m calibration.fit_churn \
    --target_annual_rate 0.15

# Fit energy model
python -m calibration.fit_energy \
    --use_literature
```

### 4. Evaluation

```bash
# Generate evaluation report
python -m evaluation.plots \
    --log_dir logs/my_experiment \
    --output_dir results/
```

## Experiments

### Baselines

1. **Fixed Pricing**: Static Three-Part Tariff (no RL)
2. **Penalty SAC**: Fixed λ weights for constraints
3. **CMDP SAC**: Primal-dual with learned multipliers (main method)

### Ablations

- NHPP vs Hawkes arrivals
- With/without allowance state in observation
- With/without overage term in churn model
- Cost model variants: energy-only vs full decomposition

### Metrics

| Metric | Description |
|--------|-------------|
| Profit | Revenue - Cost per episode |
| URLLC Violation | % hours violating 99.999% reliability |
| eMBB Violation | % hours below throughput target |
| Churn Rate | User departures per hour |
| Constraint Satisfaction | % episodes meeting all constraints |

## Technical Details

### Three-Part Tariff

For each slice s ∈ {URLLC, eMBB}:

```
Bill_i(s,t) = F_s × α_F + max(0, U_i - D_remaining) × p_s × α_p
```

Where:
- F_s: Base access fee
- D_remaining: Remaining allowance in billing cycle
- p_s: Overage price per MB
- α_F, α_p: RL-controlled price factors ∈ [0.8, 1.2]

### CMDP Formulation

```
max E[Σ γᵗ (R_t - C_t)]
s.t. E[Viol_URLLC] ≤ δ_u
     E[Viol_eMBB] ≤ δ_e
```

Solved via Lagrangian relaxation:
```
L(π, λ) = J(π) - λ_u × (E[V_u] - δ_u) - λ_e × (E[V_e] - δ_e)
```

### Channel Model

3GPP TR 38.901 UMi-Street Canyon:
- Path loss: PL = 32.4 + 21×log₁₀(d₃D) + 20×log₁₀(fc)
- Shadowing: σ_SF = 4.0 dB (LOS), 7.82 dB (NLOS)
- LOS probability: P_LOS = min(18/d, 1) × (1 - exp(-d/36)) + exp(-d/36)

### QoS Models

**URLLC (Finite Block Length)**:
```
ε = Q(√n × (C - R) / √V)
```
Where n = blocklength, C = Shannon capacity, V = channel dispersion.

**eMBB (Throughput)**:
```
Throughput = PRB_allocated × SE × BW_per_PRB
SE = f(SINR) via MCS table (3GPP 38.214)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourname2026oran,
  title={Dynamic Pricing for 5G O-RAN Network Slicing using
         Constrained Reinforcement Learning},
  author={Your Name},
  journal={IEEE Transactions on Mobile Computing},
  year={2026}
}
```

## References

1. 3GPP TS 38.101-1: NR User Equipment radio transmission and reception
2. 3GPP TR 38.901: Study on channel model for frequencies from 0.5 to 100 GHz
3. 3GPP TS 38.214: NR Physical layer procedures for data
4. Haarnoja et al., "Soft Actor-Critic," ICML 2018
5. Altman, "Constrained Markov Decision Processes," 1999
6. Fibich et al., "Optimal Three-Part Tariff Plans," Operations Research 2017

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- 3GPP for standardization documents
- O-RAN Alliance for architecture specifications
- Stable-Baselines3 team for RL implementations
