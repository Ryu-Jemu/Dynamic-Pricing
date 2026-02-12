# O-RAN 5G Single-Cell Slicing + 3-Part Tariff Pricing with SB3 SAC

Reference-grade simulation for joint network-slice pricing and PRB allocation
using Soft Actor-Critic reinforcement learning, with a **3-part tariff**
revenue model (base fee + allowance + overage pricing).

**Revision 7** — 8 improvements (R1–R8) based on v6 training evaluation analysis,
with 48 unit tests.

## Quick Start

```bash
chmod +x scripts/run.sh && ./scripts/run.sh
```

This executes the full pipeline: venv → user generation → tests → SAC training → evaluation → dashboard.

## Problem Description

A single 5G NR base station serves two network slices:

| Slice | Role | QoS Priority |
|-------|------|-------------|
| **URLLC** | Ultra-Reliable Low-Latency Communication | Strict (λ = 500K KRW) |
| **eMBB** | Enhanced Mobile Broadband | Moderate (λ = 50K KRW) |

An RL agent (SAC) makes **5 decisions each day** (1 step = 1 day):

| Action | Description | Range |
|--------|-------------|-------|
| `a[0]` | URLLC base fee F_U | [30K, 90K] KRW/cycle |
| `a[1]` | URLLC overage price p^over_U | [500, 5000] KRW/GB |
| `a[2]` | eMBB base fee F_E | [40K, 150K] KRW/cycle |
| `a[3]` | eMBB overage price p^over_E | [200, 3000] KRW/GB |
| `a[4]` | URLLC PRB share ρ_U | [0.05, 0.60] |

**Objective**: Maximize E[Σ γ^t Profit_t] where Profit = Revenue − Cost.

## Architecture

```
oran3pt/
├── config/default.yaml        # All scenario parameters (v7)
├── oran3pt/
│   ├── utils.py               # Config I/O, sigmoid, device selection
│   ├── gen_users.py           # Synthetic user CSV generation (§13)
│   ├── env.py                 # Gymnasium environment — 22D obs (§3–12)
│   ├── train.py               # SB3 SAC training + curriculum + multi-seed (§14)
│   ├── eval.py                # Evaluation + CLV computation (§15)
│   └── dashboard_app.py       # Streamlit/Matplotlib dashboard (§16)
├── scripts/run.sh             # End-to-end pipeline
├── tests/test_env.py          # 48 unit tests
├── data/                      # Generated users_init.csv
├── outputs/                   # Models, logs, plots
├── Revision_v5.md             # v7 training evaluation report
└── README.md
```

## Revenue Model: 3-Part Tariff (§5)

Each user on slice *s* pays per billing cycle *T* (30 days) [Grubb, AER 2009]:

```
Bill_{u,s} = F_s + p_s^over × max(0, D_{u,s} − Q_s)
```

| Parameter | Description | Source |
|-----------|-------------|--------|
| F_s | Base subscription fee (KRW/cycle) | Agent action |
| Q_s | Data allowance (GB/cycle) | Config (Q_U=5, Q_E=50) |
| p_s^over | Per-GB overage coefficient | Agent action |
| D_{u,s} | User's total monthly usage | Stochastic (lognormal) |

## Observation Space (§3.2)

22-dimensional normalized vector (expanded from 20D in v6):

| Index | Feature | Normalization |
|-------|---------|--------------|
| 0–1 | Active / inactive user fractions | /N_total |
| 2–3 | Previous join / churn counts | /N_total×0.05 |
| 4–5 | QoS violation probs (URLLC, eMBB) | [0, 1] |
| 6–8 | Revenue / cost / profit (normalized) | /scale |
| 9–13 | Previous action components | affine to [0, 1] |
| 14 | Billing cycle phase | (t mod T) / T |
| 15 | Episode progress | t / episode_len |
| 16 | URLLC allowance utilization | cycle_usage / (Q_U×N_U) [E4] |
| 17 | eMBB allowance utilization | cycle_usage / (Q_E×N_E) [E4] |
| 18 | URLLC load factor | L_U / C_U [E4] |
| 19 | eMBB load factor | L_E / C_E [E4] |
| **20** | **eMBB overage revenue rate** | over_rev_E / (p_over_E × N_E) [R5] |
| **21** | **Days remaining in cycle** | (T − cycle_step) / T [R5] |

## Reward (§15)

```
r = sign(profit) × log(1 + |profit| / scale)
    − smooth_penalty
    − retention_penalty
    + pop_bonus
```

| Component | Formula | Reference |
|-----------|---------|-----------|
| Log-transform | sign(p)·log1p(\|p\|/s) | [SB3 Tips] |
| Smoothing [R4] | Σ_i w_i × (a_t[i] − a_{t−1}[i])² | [Dalal 2018] |
| Retention [R2] | 2.0 × (n_churn / N_active) | [Wiewiora 2003, Gupta 2006] |
| Population [R6] | 0.1 × (N_active/N_total − 0.4) | [Mguni 2019, Zheng 2022] |

Clipped to [−2, 2].

## Training

SAC with automatic entropy tuning [Haarnoja et al., ICML 2018]:

| Hyperparameter | Value |
|----------------|-------|
| total_timesteps | 1,000,000 [R1] |
| learning_rate | 3×10⁻⁴ → 1×10⁻⁵ (linear) [E7] |
| batch_size | 256 |
| buffer_size | 200,000 |
| gamma | 0.995 |
| ent_coef | 0.5 (initial) → auto [R8] |
| n_seeds | 5 [E9] |
| Curriculum | Phase 1 (200K): no churn/join; Phase 2 (800K): full [R3] |

## Tests

```bash
python -m pytest tests/ -v
```

48 tests across 10 groups:

| Group | Tests | Focus |
|-------|-------|-------|
| T1 Env basics | 5 | Reset/step shapes, episode 720 steps |
| T2 Revenue | 2 | Non-negative revenue, overage accrual |
| T3 Market | 3 | Population conservation, join/churn ≥ 0 |
| T4 QoS | 3 | Violation in [0,1], sigmoid monotonicity |
| T5 Numerical | 5 | No NaN/Inf across seeds, obs bounds, reward clip |
| T6 Billing | 2 | Cycle length, step counter |
| T7 Utils | 3 | Lognormal fit accuracy, sigmoid stability |
| T8 Calibration | 5 | Churn/join targets, capacity, price sensitivity |
| T9 v5 Enhancements | 9 | 22D obs, load factors, CLV shaping, smoothing |
| **T10 v7 Enhancements** | **11** | **22D obs, per-dim smoothing, pop reward, curriculum** |

## Revision History

| Version | Focus | Tests | Key Change |
|---------|-------|-------|------------|
| v1 | Initial | 22 | Broken calibration (profit < 0) |
| v2 | Calibration (C1–C5) | 26 | Fixed capacity, churn, demand elasticity |
| v3 | SB3 transparency (F1) | 26 | Exposed silent import failure |
| v4 | CSVLogger crash (F5) | 26 | SAC actually trains |
| v5/v6 | 9 enhancements (E1–E9) | 37 | 20D obs, CLV shaping, multi-seed |
| **v7** | **8 improvements (R1–R8)** | **48** | **22D obs, curriculum, 1M steps, population reward** |

### v7 Improvements

| ID | Enhancement | Evidence |
|----|------------|----------|
| R1 | Training: 50K → 1M steps (restored) | [Henderson 2018, Haarnoja 2018] |
| R2 | Retention penalty: 0.15 → 2.0 | [Fader 2010, Wiewiora 2003] |
| R3 | Curriculum learning (2-phase) | [Narvekar 2020, Bengio 2009] |
| R4 | Per-dimension action smoothing | [Dalal 2018] |
| R5 | Observation: 20D → 22D | [Dulac-Arnold 2021] |
| R6 | Population-aware reward | [Mguni 2019, Zheng 2022] |
| R7 | Hierarchical action timing (design) | [Vezhnevets 2017, Bacon 2017] |
| R8 | Entropy coefficient schedule | [Zhou 2022] |

## 20 Requirements Checklist

| # | Requirement | Status |
|---|-------------|--------|
| 1 | Single-cell 5G O-RAN, two slices (URLLC/eMBB) | ✅ |
| 2 | Two stakeholders: Operator, End users | ✅ |
| 3 | Revenue: 3-part tariff = base_fee + overage | ✅ |
| 4 | Cost excludes Capex (single-cell) | ✅ |
| 5 | Objective: maximize E[Σ γ^t Profit_t] | ✅ |
| 6 | Generate initial user CSV with priors | ✅ |
| 7 | Train using SB3 SAC | ✅ |
| 8 | 5D continuous action space | ✅ |
| 9 | QoS violation penalty via cost function | ✅ |
| 10 | Traffic driven by active user count | ✅ |
| 11 | Online join/churn models | ✅ |
| 12 | Expectation-based optimization support | ✅ |
| 13 | Join and churn every step | ✅ |
| 14 | Dashboard for visualization | ✅ |
| 15 | Single shell script pipeline | ✅ |
| 16 | tqdm progress + MPS acceleration | ✅ |
| 17 | CLV integration | ✅ |
| 18 | Academic/standard grounding for every choice | ✅ |
| 19 | Complete README.md | ✅ |
| 20 | Nothing omitted | ✅ |

## References

| Tag | Source |
|-----|--------|
| [Bacon 2017] | Bacon et al., "The Option-Critic Architecture," AAAI 2017 |
| [Bengio 2009] | Bengio et al., "Curriculum Learning," ICML 2009 |
| [Dalal 2018] | Dalal et al., "Safe Exploration in Continuous Action Spaces," NeurIPS 2018 |
| [Dulac-Arnold 2021] | Dulac-Arnold et al., "Challenges of Real-World RL," JMLR 2021 |
| [Fader 2010] | Fader & Hardie, "Customer-Base Valuation," Marketing Science 2010 |
| [Grubb 2009] | Grubb, "Selling to Overconfident Consumers," AER 2009 |
| [Gupta 2006] | Gupta et al., "Modeling CLV," J. Service Research 2006 |
| [Haarnoja 2018] | Haarnoja et al., "Soft Actor-Critic," ICML 2018 |
| [Henderson 2018] | Henderson et al., "Deep RL that Matters," AAAI 2018 |
| [Mguni 2019] | Mguni et al., "Coordinating the Crowd," AAMAS 2019 |
| [Narvekar 2020] | Narvekar et al., "Curriculum Learning for RL," JMLR 2020 |
| [Nevo 2016] | Nevo et al., "Usage-Based Pricing," Econometrica 2016 |
| [Ng 1999] | Ng et al., "Policy Invariance Under Reward Transformations," ICML 1999 |
| [TS 38.104] | 3GPP TS 38.104 — NR Base Station radio |
| [TS 38.306] | 3GPP TS 38.306 — NR UE radio access capabilities |
| [Vezhnevets 2017] | Vezhnevets et al., "FeUdal Networks," ICML 2017 |
| [Wiewiora 2003] | Wiewiora et al., "Principled Methods for Advising RL Agents," ICML 2003 |
| [Zheng 2022] | Zheng et al., "The AI Economist," Science Advances 2022 |
| [Zhou 2022] | Zhou et al., "Revisiting Exploration in Deep RL," ICLR 2022 |

## License

Research / educational use. Not for production deployment.
