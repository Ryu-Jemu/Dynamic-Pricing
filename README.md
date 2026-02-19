# O-RAN 5G Single-Cell Slicing + 3-Part Tariff Pricing with SB3 SAC

Reference-grade simulation for joint network-slice pricing and PRB allocation
using Soft Actor-Critic reinforcement learning, with a **3-part tariff**
revenue model (base fee + allowance + overage pricing).

**Revision 5.7** — Consolidated from v10.4 + v5.7 RESOLUTION_PLAN,
with 183 unit tests.

## Quick Start

```bash
chmod +x scripts/run.sh && ./scripts/run.sh
```

This executes the full pipeline: venv → user generation → tests → SAC training → evaluation → dashboard.

Interactive mode prompts for hyperparameters (timesteps, seeds, buffer, batch, lr).
Use `--auto` to skip prompts and use config defaults.

## Problem Description

A single 5G NR base station serves two network slices:

| Slice | Role | QoS Priority |
|-------|------|-------------|
| **URLLC** | Ultra-Reliable Low-Latency Communication | Strict (λ = 500K KRW) |
| **eMBB** | Enhanced Mobile Broadband | Strict (λ = 500K KRW, γ=3.0 convex) |

An RL agent (SAC) makes **5 decisions each day** (1 step = 1 day):

| Action | Description | Range |
|--------|-------------|-------|
| `a[0]` | URLLC base fee F_U | [30K, 90K] KRW/cycle |
| `a[1]` | URLLC overage price p^over_U | [500, 5000] KRW/GB |
| `a[2]` | eMBB base fee F_E | [35K, 110K] KRW/cycle |
| `a[3]` | eMBB overage price p^over_E | [500, 3000] KRW/GB |
| `a[4]` | URLLC PRB share ρ_U | [0.03, 0.10] |

**Objective**: Maximize E[Σ γ^t Profit_t] where Profit = Revenue − Cost,
subject to E[pviol_E] ≤ ε_QoS (Constrained MDP, Lagrangian dual ascent).

## O-RAN Architecture Mapping

The SAC agent is deployed as an **rApp** on the **Non-RT RIC** (within SMO),
consistent with O-RAN Alliance architecture specifications:

| O-RAN Component | Role in This System | Decision Timescale | Standard |
|-----------------|--------------------|--------------------|----------|
| **Non-RT RIC (SMO)** | SAC rApp: pricing + ρ_U policy | 1 day (ρ_U), 30 days (pricing) | O-RAN WG1 OAD §5.1 |
| **A1 Interface** | Policy guidelines delivery | Per billing cycle | O-RAN WG1 OAD §6.2 |
| **Near-RT RIC** | Enforcement: admission control, PRB limit translation | Simulated daily | O-RAN WG3 RICARCH §4.1 |
| **E2 Interface** | Resource limits to O-DU | Per ρ_U update | O-RAN WG3 RICARCH §5.2 |
| **O-DU / gNB** | MAC scheduler (rule-based, abstracted) | 1 ms TTI (aggregated) | 3GPP TS 38.321 §5.4 |

> **Simulation abstraction**: The Near-RT RIC enforcement and O-DU MAC scheduler
> are modeled within the Gymnasium environment (`env.py`) as capacity allocation
> formulas (`C_s = ρ_s × C_total`) and sigmoid-based QoS violation functions.
> TTI-level scheduling is aggregated to daily load statistics.
> [Polese IEEE CST 2023 §III-B; Bonati IEEE TMC 2023 §II-A]

## Architecture

```
Dynamic-Pricing/
├── config/
│   ├── default.yaml              # All scenario parameters (v10.4)
│   └── production.yaml           # Production override (1M steps, 5 seeds)
├── oran3pt/
│   ├── utils.py                  # Config I/O, sigmoid, device selection
│   ├── gen_users.py              # Synthetic user CSV generation
│   ├── env.py                    # Gymnasium environment — 23D obs, 5D action
│   ├── train.py                  # SB3 SAC training + curriculum + multi-seed
│   ├── eval.py                   # Evaluation + CLV computation + dashboards
│   ├── dashboard_app.py          # Streamlit / Matplotlib dashboard
│   ├── html_dashboard.py         # HTML convergence dashboard (Chart.js)
│   ├── png_dashboard.py          # 7-sheet PNG dashboard (28 panels)
│   ├── business_metrics.py       # Business KPI computation
│   ├── business_dashboard.py     # Business dashboard (Chart.js)
│   └── templates/                # HTML templates
├── scripts/run.sh                # End-to-end pipeline (interactive)
├── tests/test_env.py             # 183 unit tests across 27 groups
├── data/                         # Generated users_init.csv
├── outputs/                      # Models, logs, plots, dashboards
├── docs/                         # Consolidated documentation
│   └── PROJECT_KNOWLEDGE.md      # Unified project knowledge base
└── README.md
```

## Revenue Model: 3-Part Tariff

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

## Observation Space (23-D)

23-dimensional normalized vector [ME-1]:

| Index | Feature | Normalization |
|-------|---------|--------------|
| 0 | Active user fraction | N_active / N_total |
| 1 | Normalized joins | n_join / (N_total × 0.05) |
| 2 | Normalized churns | n_churn / (N_total × 0.05) |
| 3 | URLLC QoS violation | pviol_U ∈ [0, 1] |
| 4 | eMBB QoS violation | pviol_E ∈ [0, 1] |
| 5 | Normalized revenue | revenue / scale |
| 6 | Normalized cost | cost / scale |
| 7 | Normalized profit | profit / scale |
| 8–12 | Previous action (5D) | affine to [0, 1] |
| 13 | Billing cycle phase | (t mod T) / T |
| 14 | Churn rate EMA [EP1] | churn_rate_ema |
| 15 | URLLC allowance utilization | cycle_usage_U / (Q_U × N_U) |
| 16 | eMBB allowance utilization | cycle_usage_E / (Q_E × N_E) |
| 17 | URLLC load factor | L_U / C_U |
| 18 | eMBB load factor | L_E / C_E |
| 19 | eMBB overage revenue rate | over_rev_E / (p_over_E × N_E) |
| 20 | Days remaining in cycle | (T − cycle_step) / T |
| 21 | pviol_E EMA [D6] | EMA(α=0.3) ∈ [0, 1] |
| 22 | eMBB load headroom [D5] | max(0, 1 − L_E / C_E) |

## Reward

```
r_base = clip(sign(profit) × log(1 + |profit| / 300K)
              − smooth_penalty − retention_penalty + pop_bonus, −2, 2)
r_final = clip(r_base − lagrangian_penalty × boost, −8, 8)
```

| Component | Formula | Reference |
|-----------|---------|-----------|
| Log-profit | sign(p)·log1p(\|p\|/300K) | [SB3 Tips] |
| Smoothing [R4] | Σ_i w_i × (a_t[i] − a_{t−1}[i])² | [Dalal 2018] |
| Retention [R2] | 15.0 × (n_churn / N_active) | [Wiewiora 2003, Fader 2010] |
| Population [R6] | 2.0 × quadratic(N_active/N_total − 0.37) | [Mguni 2019, Zheng 2022] |
| Lagrangian [D2] | λ × pviol_E × boost | [Tessler 2019, Stooke 2020] |
| Capacity Guard | 4.0 × max(0, L_E/C_E − 0.78)² | [Samdanis 2016] |

## Training

SAC with automatic entropy tuning [Haarnoja et al., ICML 2018]:

| Hyperparameter | Value |
|----------------|-------|
| total_timesteps | 1,000,000 [R1] |
| learning_rate | 3×10⁻⁴ → 1×10⁻⁵ (linear) [E7] |
| batch_size | 512 [OPT-D] |
| buffer_size | 200,000 |
| gamma | 0.995 |
| ent_coef | auto_0.5 [R8] |
| n_seeds | 5 (parallel) [E9] |
| Curriculum | 3-phase: P1 (10%, no churn) → P2 (35%, QoS focus) → P3 (55%, full) |
| Early stopping | patience=15, min_timesteps=auto (75%) [OPT-C] |
| PID Lagrangian | Kp=0.05, Ki=0.015, Kd=0.01, λ_max=15 [D2] |

## Tests

```bash
python -m pytest tests/ -v
```

183 tests across 27 groups (T1–T27, no T14/T19):

| Group | Tests | Focus |
|-------|-------|-------|
| T1 Env basics | 5 | Reset/step shapes, episode 30 steps |
| T2 Revenue | 2 | Non-negative revenue, overage accrual |
| T3 Market | 3 | Population conservation, join/churn ≥ 0 |
| T4 QoS | 3 | Violation in [0,1], sigmoid monotonicity |
| T5 Numerical | 5 | No NaN/Inf across seeds, obs bounds, reward clip |
| T6 Billing | 2 | Cycle length, step counter |
| T7 Utils | 3 | Lognormal fit accuracy, sigmoid stability |
| T8 Calibration | 4 | Churn/join targets, capacity |
| T9 v5 enhancements | 4 | 23D obs, load factors, CLV shaping |
| T10 v7 enhancements | 4 | Per-dim smoothing, pop reward, curriculum |
| T11 v8 enhancements | 8 | Curriculum, convex SLA, Lagrangian, smoothing |
| T12 v9 design | 10 | Admission, hierarchical, 23D obs, backward compat |
| T13 Dashboard smoke | 3 | PNG dashboard import, episode detection |
| T15 Design D1-D7 | 7 | Slice-specific Q_sig, PID Lagrangian, pviol_E EMA |
| T16 EP1 continuous | 7 | Truncation, population persistence, global counter |
| T17 Business dashboard | 12 | KPI keys, P&L, revenue %, SLA range |
| T18 OPT training | 6 | Entropy, early stop, parallel, cache |
| T19 M15 dashboards | 4 | Imports, params, graceful missing |
| T20 Architecture review | 8 | Anti-windup, 23D obs, config merge, integration |
| T21 V11 improvements | 9 | rho_U_max, beta_pop, admission, Lagrangian state |
| T22 PR pricing | 11 | Per-slice P_sig, bill shock, overage join |
| T23 v11.1 structural | 18 | Asymmetric integral, lambda_min, capacity guard |
| T24 v5.3 fixes | 10 | Eval λ propagation, PID integral, rho_U bounds |
| T25 v5.5 improvements | 8 | Lambda_max, threshold, boost decay, DR |
| T26 v5.6 corrections | 15 | Alpha_congestion, AC, reward_scale, capacity |
| T27 v5.7 RESOLUTION_PLAN | 12 | Shaping recalib, PID gain scheduling, constraint selection |

## Revision History

| Version | Focus | Tests | Key Change |
|---------|-------|-------|------------|
| v1 | Initial | 22 | Broken calibration (profit < 0) |
| v2 | Calibration (C1–C5) | 26 | Fixed capacity, churn, demand elasticity |
| v3 | SB3 transparency (F1) | 26 | Exposed silent import failure |
| v4 | CSVLogger crash (F5) | 26 | SAC actually trains |
| v5/v6 | 9 enhancements (E1–E9) | 37 | 20D obs, CLV shaping, multi-seed |
| v7 | 8 improvements (R1–R8) | 48 | 22D obs, curriculum, 1M steps |
| v8 | Infra + dashboards (M1–M15) | 96 | PNG/HTML/business dashboards, Lagrangian |
| v9 | Design improvements (D1–D7) | 100 | PID Lagrangian, slice-specific Q_sig, 3-phase |
| v10 | EP1 + review (CR/ME/HI) | 109 | 1-cycle continuous, 23D obs, anti-windup |
| v11 | Market + pricing (V11/PR) | 136 | rho_U bounds, bill shock, per-slice P_sig |
| v10.4 | Hotfixes (v5.2–v5.6) | 171 | α=15, λ_max=5, AC 0.80, reward calibration |
| **v5.7** | **RESOLUTION_PLAN** | **183** | **λ_max=15, clip [-8,8], PID gain scheduling, constraint-aware selection** |

## References

| Tag | Source |
|-----|--------|
| [Achiam 2017] | Achiam et al., "Constrained Policy Optimization," ICML 2017 |
| [Bacon 2017] | Bacon et al., "The Option-Critic Architecture," AAAI 2017 |
| [Bengio 2009] | Bengio et al., "Curriculum Learning," ICML 2009 |
| [Bonati 2023] | Bonati et al., "ColO-RAN: ML-based xApps for Open RAN," IEEE TMC 2023 |
| [Boyd 2004] | Boyd & Vandenberghe, "Convex Optimization," Cambridge 2004 |
| [Dalal 2018] | Dalal et al., "Safe Exploration in Continuous Action Spaces," NeurIPS 2018 |
| [Dulac-Arnold 2021] | Dulac-Arnold et al., "Challenges of Real-World RL," JMLR 2021 |
| [Fader 2010] | Fader & Hardie, "Customer-Base Valuation," Marketing Science 2010 |
| [Grubb 2009] | Grubb, "Selling to Overconfident Consumers," AER 2009 |
| [Grubb & Osborne 2015] | Grubb & Osborne, "Cellular Service Demand: Bill Shock," AER 2015 |
| [Gupta 2006] | Gupta et al., "Modeling CLV," J. Service Research 2006 |
| [Haarnoja 2018] | Haarnoja et al., "Soft Actor-Critic," ICML 2018 |
| [Henderson 2018] | Henderson et al., "Deep RL that Matters," AAAI 2018 |
| [Lambrecht & Skiera 2006] | Lambrecht & Skiera, "Paying Too Much and Being Happy About It," JMR 2006 |
| [Mguni 2019] | Mguni et al., "Coordinating the Crowd," AAMAS 2019 |
| [Nahum 2024] | Nahum et al., "Intent-Aware Radio Resource Management for DRL-based Network Slicing," IEEE TMC 2024 |
| [Narvekar 2020] | Narvekar et al., "Curriculum Learning for RL," JMLR 2020 |
| [Nevo 2016] | Nevo et al., "Usage-Based Pricing," Econometrica 2016 |
| [Ng 1999] | Ng et al., "Policy Invariance Under Reward Transformations," ICML 1999 |
| [O-RAN WG1] | O-RAN Alliance, "O-RAN Architecture Description," O-RAN.WG1.OAD-R003-v10.00, 2023 |
| [O-RAN WG2] | O-RAN Alliance, "AI/ML Workflow Description," O-RAN.WG2.AIML-v01.03, 2023 |
| [O-RAN WG3] | O-RAN Alliance, "Near-RT RIC Architecture," O-RAN.WG3.RICARCH-R003-v04.00, 2023 |
| [Pardo 2018] | Pardo et al., "Time Limits in Reinforcement Learning," ICML 2018 |
| [Polese 2023] | Polese et al., "Understanding O-RAN," IEEE Comms Surveys & Tutorials 2023 |
| [Raffin 2021] | Raffin et al., "Stable-Baselines3: Reliable RL Implementations," JMLR 2021 |
| [Saha 2023] | Saha et al., "DRL Approaches to Network Slice Scaling: A Survey," IEEE CommMag 2023 |
| [Stooke 2020] | Stooke et al., "Responsive Safety in RL by PID Lagrangian Methods," ICLR 2020 |
| [Sulaiman 2023] | Sulaiman et al., "Coordinated Slicing and Admission Control Using Multi-Agent DRL," IEEE TNSM 2023 |
| [Tessler 2019] | Tessler et al., "Reward Constrained Policy Optimization," ICML 2019 |
| [Tirole 1988] | Tirole, "The Theory of Industrial Organization," MIT Press 1988 |
| [Train 2009] | Train, "Discrete Choice Methods with Simulation," Cambridge 2009 |
| [TS 38.104] | 3GPP TS 38.104 — NR Base Station radio |
| [TS 38.306] | 3GPP TS 38.306 — NR UE radio access capabilities |
| [Vezhnevets 2017] | Vezhnevets et al., "FeUdal Networks," ICML 2017 |
| [Wan 2025] | Wan et al., "Empirical Study of Deep RL in Continuing Tasks," arXiv 2025 |
| [Wiewiora 2003] | Wiewiora et al., "Principled Methods for Advising RL Agents," ICML 2003 |
| [Zheng 2022] | Zheng et al., "The AI Economist," Science Advances 2022 |
| [Zhou 2022] | Zhou et al., "Revisiting Exploration in Deep RL," ICLR 2022 |

## License

Research / educational use. Not for production deployment.
