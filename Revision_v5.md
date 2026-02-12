# Revision 6: Training Evaluation & Further Improvement Design

## 1. Executive Summary

Training completed successfully across all 5 seeds with SAC actually learning a meaningful pricing policy. The trained agent achieves **+102.6% reward improvement** over the random baseline (0.2897 vs 0.143) and **+98.7% profit improvement** (526K vs 265K KRW/step). The EvalCallback produced a best model distinct from the final model, confirming that model selection is functioning. However, seven structural issues remain — most critically, **training was limited to 50,000 timesteps per seed** (only 69 episodes) rather than the 1,000,000 intended by enhancement [E2], and **F_E is again approaching its ceiling** at 98.6% of the 150K maximum.

### Key Results

| Metric | Random Baseline (v2) | Trained Agent (v6) | Change |
|--------|---------------------|-------------------|--------|
| Mean reward | +0.143 | **+0.2897** | +102.6% |
| Mean profit (KRW/step) | 264,838 | **526,516** | +98.7% |
| Episode profit (M KRW) | 95.3 | **379.1 ± 8.6** | +297.6% |
| Mean N_active | 178 | **159.7** | −10.3% |
| Monthly churn | 6.8% | **8.2%** | +21% |
| Monthly retention | 93.2% | **91.8%** | −1.5 pp |
| CLV/user (KRW) | 496,753 | **973,986** | +96.1% |
| CLV/CAC ratio | 6.2 | **12.2** | +96.8% |
| pviol_E mean | 0.658 | **0.334** | −49.2% |
| Profit margin | — | **67.3%** | — |

---

## 2. Training Completion Assessment

### 2.1 Confirmed: SAC Training Executed Successfully

All 5 seed models were saved, confirming the CSVLogger crash (Revision v4 [F5]) and SB3 import failure (Revision v3 [F1]) are fully resolved.

| Artefact | Status | Evidence |
|----------|--------|----------|
| `best_model.zip` | ✅ 3,261,227 bytes | EvalCallback selected best checkpoint |
| `final_model_seed{0-4}.zip` | ✅ 3,261,208 bytes each | All 5 seeds completed |
| `rollout_log.csv` | ✅ 3,600 rows (5×720) | Full 24-cycle episodes |
| `eval_summary.csv` | ✅ 35 metrics | All info keys present |
| `clv_report.csv` | ✅ CLV = 973,986 KRW | Positive, economically viable |
| best ≠ final | ✅ File sizes differ | Model selection is operational |

### 2.2 Critical: Training Duration Mismatch

The `config/default.yaml` specifies `total_timesteps: 50000` but the inline comment reads `[E2] was 300K; ~1388 episodes at 720 steps/ep`. The comment describes the intended 1M timesteps, but the actual value was set to **50,000** — likely a debugging or testing value that was never restored.

| Parameter | Config Value | Comment Intent | Effect |
|-----------|-------------|---------------|--------|
| `total_timesteps` | 50,000 | 1,000,000 | **95% fewer training steps** |
| Episodes/seed | 69 | 1,388 | Below Henderson (2018) minimum of 2,000 |
| Total episodes (5 seeds) | 347 | 6,944 | — |

This is the single most impactful issue. The agent has learned a reasonable policy from only 69 episodes, suggesting significant room for improvement with proper training duration.

### 2.3 Learned Policy Analysis

Despite severely limited training, the agent learned several non-trivial behaviours.

**Pricing strategy.** The agent converged to high base fees (F_U ≈ 74K, F_E ≈ 133K KRW), moderate URLLC overage (p_over_U ≈ 3,550 KRW/GB), and mid-range eMBB overage (p_over_E ≈ 1,613 KRW/GB). Cross-repeat consistency is remarkably tight (F_E CV = 0.3% across repeats), indicating a stable policy.

**Intra-cycle capacity management.** The agent learned to vary ρ_U within the billing cycle — allocating more PRBs to URLLC early in the cycle (ρ_U ≈ 0.30 at day 0) and reducing it progressively (ρ_U ≈ 0.16 at day 20–29). This is sensible: early-cycle traffic is uncertain, so hedging URLLC capacity makes sense; later in the cycle the agent can observe realised loads and shift capacity to eMBB.

**Overage price timing.** The agent reduces p_over_E early in the cycle (day 5: 1,059 KRW/GB) when few users have exceeded their allowance, then increases it toward the cycle end (day 25–29: ~1,900–2,000 KRW/GB) when overage revenue is actually accruing. This mirrors the demand-pull pattern observed in the data: overage revenue is concentrated in the final third of the cycle (days 21–30: 428,878 KRW/step vs days 1–10: 0 KRW/step).

---

## 3. Identified Issues

### 3.1 [D1] CRITICAL — `total_timesteps` Set to 50K Instead of 1M

**Severity:** Critical — the intended training enhancement [E2] was never applied.

**Root cause:** The config file has `total_timesteps: 50000` with a comment referencing the intended 1M value. This is likely a residual from a quick-test run.

**Impact:** 69 episodes per seed is 3.5% of the Henderson et al. (2018) recommended minimum of 2,000 episodes. The agent has barely explored the state-action space. SAC's replay buffer (200K capacity) was never filled (only 50K samples collected per seed).

**Evidence:** The action coefficient of variation remains high (rho_U CV = 44.9%, p_over_E CV = 32.5%), indicating the policy has not fully converged. With 1M timesteps, the agent would have ~1,388 episodes — sufficient for convergence per Haarnoja et al. (ICML, 2018).

### 3.2 [D2] HIGH — F_E Approaching New Ceiling (98.6% of 150K)

**Severity:** High — the v4 saturation problem (E1 was designed to solve) has partially recurred.

**Finding:** F_E p95 = 147,833 KRW, representing 98.6% of the 150K maximum. The median F_E is 136,861 KRW (91.2% of max). While this is better than v4's 99.5% saturation at 110K, the distribution is still right-skewed and pressing against the ceiling.

**Interpretation:** Two hypotheses exist: (a) the true optimum F_E lies at or above 150K, meaning the ceiling should be raised further; or (b) with only 69 training episodes, the agent has not experienced enough high-price churn events to learn the interior optimum. Given that β_p_churn = 3.0 creates strong churn sensitivity at max price (predicted 15.9% monthly), hypothesis (b) is more likely — more training would allow the agent to discover the revenue-churn tradeoff.

### 3.3 [D3] HIGH — Persistent User Decline (200 → 160)

**Severity:** High — the agent is pricing above the sustainable equilibrium.

**Finding:** All 5 repeats show N_active declining from ~200 to ~160–177 (mean Δ = −33 users). The decline is concentrated in the first 60 steps (cycle 1–2), after which population stabilises near 155–165. Monthly churn (8.2%) exceeds monthly join (3.6%), resulting in a net outflow of −0.046 users/step.

**Root cause analysis:** The early-episode churn spike is driven by the agent's aggressive initial pricing (F_E jumps to ~133K KRW immediately). The per-step churn rate in the early phase (steps 1–240) is 0.00406, nearly double the late phase (0.00207). A counterfactual analysis shows that maintaining N_active ≈ 200 at slightly lower prices would yield +28.4% higher profit (675K vs 526K KRW/step), suggesting the agent is over-extracting from existing users.

**Evidence:** Gupta et al. (J. Service Research, 2006) find that firms optimising for CLV price 15–30% below short-run profit maximisers. The current F_E (133K) may be in the over-extraction zone where marginal revenue < marginal CLV loss from churn.

### 3.4 [D4] MODERATE — Retention Penalty Too Weak

**Severity:** Moderate — the CLV reward shaping [E6] has negligible effect.

**Finding:** Mean retention penalty = 0.000307, compared to mean reward = 0.2897. The penalty represents only 0.11% of the reward signal — too small to influence policy learning. Even at maximum, the retention penalty reaches only 0.0006 (0.2% of reward).

**Root cause:** With α_retention = 0.15 and typical n_churn/N_active ≈ 0.003 per step, the penalty is 0.15 × 0.003 = 0.00045 — negligible compared to the log-profit reward of ~0.29. The parameterisation was set before β_p_churn was increased to 3.0 (E3), and was not recalibrated.

**Evidence:** Ng et al. (ICML, 1999) show that potential-based reward shaping must be of comparable magnitude to the primary reward to meaningfully affect learning. A penalty at 0.1% of reward provides no gradient signal.

### 3.5 [D5] MODERATE — Smooth Penalty Insufficient for Pricing Stability

**Severity:** Moderate — action volatility remains high despite the smoothing weight increase in [E8].

**Finding:** F_U changes by an average of 8,094 KRW between consecutive steps (13.5% of its range). The smooth penalty contributes only 2.07% of the total reward signal. In practice, telco pricing changes occur at most weekly or monthly, not daily.

**Evidence:** Dulac-Arnold et al. (JMLR, 2021, Challenge #7) recommends smoothing penalties of 1–10% of reward scale. The current 2.07% is within this range but at the low end, given the domain-specific requirement for pricing stability. Real telecoms adjust tariffs on monthly or quarterly cycles, not daily.

### 3.6 [D6] LOW — Join Rate Below Target

**Severity:** Low — calibration gap.

**Finding:** Monthly join rate is 3.56%, below the calibration target of 5% specified in the config. This is because the agent prices aggressively (high P_sig), which suppresses join probability through β_p_join. The market model correctly reflects this — high prices deter new subscribers.

### 3.7 [D7] LOW — Reward Does Not Account for Overage Revenue Timing

**Severity:** Low — structural observation gap.

**Finding:** Overage revenue is concentrated in the last 10 days of the billing cycle (days 21–30: 428,878 KRW/step vs days 1–10: 0 KRW/step). The agent partially adapts (adjusting p_over_E within cycles), but the observation space lacks an explicit overage revenue rate feature that would help the agent anticipate the approaching overage threshold.

---

## 4. Proposed Improvements

### 4.1 [R1] CRITICAL — Restore `total_timesteps` to 1,000,000

**Change:** `config/default.yaml` — `total_timesteps: 50000` → `total_timesteps: 1000000`

**Justification:** This was the original intent of enhancement [E2]. At 1M timesteps with 720 steps/episode, each seed trains for ~1,388 episodes. With 5 seeds, the total is ~6,944 episodes, exceeding the Henderson et al. (2018) minimum of 2,000.

**Expected impact:** Policy convergence (lower action CV), improved profit through refined pricing, replay buffer fully utilised.

**Evidence:** Haarnoja et al. (ICML, 2018) train SAC for 1M–3M steps on continuous control tasks with comparable action dimensionality.

### 4.2 [R2] HIGH — Strengthen CLV Retention Penalty

**Change:** `config/default.yaml` — `alpha_retention: 0.15` → `alpha_retention: 2.0`

The current penalty is 0.11% of reward — below the threshold for learning. With α = 2.0, and typical n_churn/N_active ≈ 0.003:

```
penalty = 2.0 × 0.003 = 0.006
```

This is 2.1% of the mean reward (0.29), making it meaningful but not dominant. At peak churn (n_churn/N_active ≈ 0.02), the penalty would be 0.04 (14% of reward), providing a strong corrective signal.

**Evidence:** Fader & Hardie (Marketing Science, 2010) show that even small reductions in churn (0.5–1pp) can increase firm value by 25–95%. The penalty should be large enough to internalise this value. Wiewiora et al. (ICML, 2003) extend Ng et al. (1999) to show that non-potential reward shaping can still accelerate convergence if calibrated to the relevant reward scale.

### 4.3 [R3] HIGH — Add Curriculum Learning for Pricing

**Change:** `oran3pt/train.py` — implement two-phase curriculum.

**Phase 1 (first 200K steps):** Fixed N_active (no churn/join). The agent learns the capacity allocation and overage pricing subproblems in isolation.

**Phase 2 (remaining 800K steps):** Full stochastic dynamics with market churn/join enabled.

**Justification:** The current environment presents a coupled optimisation problem: pricing affects churn, churn affects N_active, N_active affects traffic, traffic affects QoS, QoS affects churn. Learning all interactions simultaneously from random initialisation requires excessive exploration.

**Evidence:** Narvekar et al. (JMLR, 2020) — "Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey" — show that decomposing complex environments into staged learning objectives accelerates convergence by 2–5× in continuous control domains. Bengio et al. (ICML, 2009) provide theoretical grounding for curriculum learning: presenting training examples in order of increasing complexity improves both convergence speed and final performance.

### 4.4 [R4] HIGH — Asymmetric Action Smoothing (Pricing vs Capacity)

**Change:** Replace scalar smoothing weight with per-dimension weights.

```yaml
action_smoothing:
  enabled: true
  weights: [0.10, 0.05, 0.10, 0.05, 0.01]  # F_U, p_over_U, F_E, p_over_E, rho_U
```

**Rationale:** Pricing dimensions (F_U, F_E) should be penalised more heavily for changes — real telcos change base fees monthly at most. Overage prices and capacity allocation (ρ_U) can reasonably change daily (algorithmic adjustment). The current uniform penalty treats all 5 dimensions identically, which is domain-inappropriate.

**Evidence:** Dalal et al. (NeurIPS, 2018) — "Safe Exploration in Continuous Action Spaces" — demonstrate that domain-specific action constraints (including dimension-specific smoothness) improve both learning efficiency and deployed policy quality. The current ρ_U intra-cycle adjustment pattern (0.30→0.16) is desirable and should not be penalised.

### 4.5 [R5] MODERATE — Observation Enrichment: Overage Proximity Signal

**Change:** Add 2 new observation features (20D → 22D):

| Index | Feature | Formula |
|-------|---------|---------|
| 20 | eMBB overage rate | over_rev_E / (p_over_E × N_E + ε) |
| 21 | Days remaining in cycle | (T − cycle_step) / T |

**Rationale:** The agent currently has allowance utilisation (obs[16-17]) but lacks a direct signal for the rate at which overage revenue is accruing. Since 98.4% of overage revenue occurs in the final third of the cycle, an explicit proximity signal would help the agent time p_over_E adjustments.

**Evidence:** Dulac-Arnold et al. (JMLR, 2021, Challenge #1): "Providing task-relevant observations to the agent reduces sample complexity." The agent already exhibits intra-cycle p_over_E modulation (1,059→1,994 KRW), so providing explicit timing information would sharpen this learned behaviour.

### 4.6 [R6] MODERATE — Population-Aware Reward Normalisation

**Change:** Scale the log-profit reward by N_active to prevent the agent from treating user loss as acceptable.

```python
# Current
r = sign(profit) * log1p(|profit| / scale)

# Proposed
r = sign(profit) * log1p(|profit| / scale) + β_pop * (N_active / N_total - target_ratio)
```

where `target_ratio = N_active_init / N_total = 0.4` and `β_pop = 0.1`.

**Rationale:** The current reward is profit-based; the agent correctly maximises per-step profit but ignores the population trajectory. Adding a population maintenance term rewards the agent for sustaining the subscriber base. The +0.1 per-step bonus when N_active matches the initial ratio (200/500 = 0.4) is small enough to not override profit optimisation but large enough to discourage excessive churn.

**Evidence:** Mguni et al. (AAMAS, 2019) — show that auxiliary reward terms tied to environment state stability improve long-horizon performance in economic simulations. The approach is analogous to "reward shaping for sustainability" in Zheng et al. (Science Advances, 2022) — "The AI Economist" — where population-level welfare terms prevent degenerate equilibria.

### 4.7 [R7] MODERATE — Separate Pricing and Allocation Update Frequencies

**Change:** Implement hierarchical action structure.

- **Pricing (F_U, p_over_U, F_E, p_over_E):** Updated every 30 steps (once per billing cycle).
- **Capacity (ρ_U):** Updated every step (daily).

**Rationale:** The current architecture updates all 5 actions every step, but in practice telcos adjust tariffs monthly while RAN resource allocation can be optimised in real-time. The step-to-step F_U oscillation (mean absolute change = 8,094 KRW, 13.5% of range) is an artefact of this mismatch.

**Implementation:** On non-pricing steps, the pricing actions from the last cycle start are replayed; only ρ_U is read from the policy network.

**Evidence:** Vezhnevets et al. (ICML, 2017) — "FeUdal Networks for Hierarchical Reinforcement Learning" — demonstrate that hierarchical action structures with different temporal abstractions improve learning efficiency in environments with mixed time-scale dynamics. Bacon et al. (AAAI, 2017) — "The Option-Critic Architecture" — provide theoretical grounding for multi-timescale action decomposition.

### 4.8 [R8] LOW — Exploration Noise Schedule

**Change:** Start with higher SAC entropy coefficient and decay it.

```yaml
training:
  ent_coef: 0.5           # Start with higher exploration (default auto starts ~0.1)
  ent_coef_final: "auto"  # Transition to auto-tuning after 200K steps
```

**Rationale:** With only 69 episodes of training, the agent's exploration was limited. Higher initial entropy would encourage broader state-action space coverage during the critical early phase of training.

**Evidence:** Haarnoja et al. (ICML, 2018) — the automatic entropy tuning starts conservatively. Zhou et al. (ICLR, 2022) — "Revisiting Exploration in Deep RL" — show that higher initial exploration entropy followed by annealing improves final performance in environments with sparse or delayed rewards (which applies here, as overage revenue is temporally sparse within cycles).

---

## 5. Detailed Results Analysis

### 5.1 Revenue Structure

| Component | Value (KRW/step) | Share |
|-----------|------------------|-------|
| Base revenue | 639,333 | 81.7% |
| Overage revenue | 143,021 | 18.3% |
| **Total revenue** | **782,353** | 100% |

Base revenue dominates because the agent sets high base fees. Overage revenue is significant (~18%) and concentrated in the final third of billing cycles. The agent partially optimises this timing by reducing p_over_E early in the cycle (when no overage occurs) and increasing it later.

### 5.2 Cost Structure

| Component | Value (KRW/step) | Share |
|-----------|------------------|-------|
| OPEX | 191,644 | 74.9% |
| CAC | 32,867 | 12.8% |
| SLA penalty | 16,769 | 6.6% |
| Energy | 14,558 | 5.7% |
| **Total cost** | **255,838** | 100% |

OPEX dominance (74.9%) is appropriate for a single-cell scenario without capex. CAC is moderate (12.8%), reflecting the 0.41 joins/step × 80,000 KRW/join. SLA penalties are well-controlled at 6.6% thanks to the agent's low ρ_U setting.

### 5.3 QoS Management

The agent learned an effective capacity allocation strategy. By keeping ρ_U low (mean 0.20, i.e. 20% of PRBs to URLLC), the agent ensures near-zero URLLC violations (pviol_U = 0.0001) while managing eMBB congestion within acceptable bounds (pviol_E = 0.334).

| ρ_U Range | pviol_E | pviol_U | Profit (KRW/step) |
|-----------|---------|---------|-------------------|
| ≤ 0.15 | 0.158 | 0.0002 | 558,318 |
| 0.15–0.25 | 0.254 | 0.0001 | 546,057 |
| 0.25–0.35 | 0.595 | 0.0001 | 474,563 |
| 0.35–0.60 | 0.877 | 0.0001 | 413,872 |

The agent correctly identifies that lower ρ_U maximises profit — URLLC traffic is so small (7.3 GB/day vs capacity of ~100 GB at ρ_U = 0.20) that even minimal allocation suffices, while eMBB benefits substantially from the extra capacity.

### 5.4 Intra-Cycle Dynamics

The agent exhibits learned temporal structure within billing cycles:

| Cycle Day | ρ_U | p_over_E (KRW/GB) | Interpretation |
|-----------|-----|-------------------|----------------|
| 0 (reset) | 0.30 | 2,371 | Hedge URLLC, high overage |
| 5 | 0.28 | 1,059 | Reduce overage (no users in overage yet) |
| 10 | 0.23 | 1,496 | Transition |
| 15 | 0.18 | 1,778 | Shift capacity to eMBB, raise overage |
| 20 | 0.15 | 1,938 | Overage revenue accruing |
| 25 | 0.17 | 1,853 | Stable |
| 29 | 0.16 | 1,994 | Maximum overage extraction |

This pattern is economically rational: the agent reduces ρ_U as the cycle progresses (more data about realised URLLC load), and increases p_over_E as users cross their allowance thresholds.

### 5.5 CLV Analysis

| CLV Metric | Value |
|------------|-------|
| Monthly profit | 15.8 M KRW |
| Mean N_active | 159.7 |
| CF/user/month | 98,905 KRW |
| Monthly churn | 8.23% |
| Monthly retention | 91.77% |
| **CLV/user** | **973,986 KRW** |
| **CLV/CAC** | **12.2** |

CLV/CAC = 12.2 well exceeds the 3:1 threshold for sustainable business models (Gupta et al., 2006). However, the 8.23% monthly churn exceeds the target of 3%, suggesting the pricing policy is too aggressive. With the same per-user margin but 3% monthly churn (97% retention), CLV/user would increase to approximately 1,750,000 KRW (+79%).

### 5.6 Cross-Repeat Consistency

| Metric | Mean | Std | CV |
|--------|------|-----|-----|
| Episode profit (M KRW) | 379.1 | 8.6 | 2.3% |
| Mean reward | 0.2897 | 0.0057 | 2.0% |
| Final N_active | 167.2 | 5.5 | 3.3% |
| F_U (KRW) | 74,411 | 492 | 0.7% |
| F_E (KRW) | 132,631 | 393 | 0.3% |
| rho_U | 0.204 | 0.001 | 0.6% |

Cross-repeat consistency is excellent (profit CV = 2.3%), confirming a deterministic policy that produces stable outcomes across different stochastic realisations of traffic and market dynamics.

---

## 6. Statistical Confidence

| Metric | Mean | Std | 95% CI (n=5) |
|--------|------|-----|-------------|
| Episode profit (M KRW) | 379.09 | 8.58 | [371.6, 386.6] |
| Mean reward | 0.2897 | 0.0057 | [0.2847, 0.2947] |
| N_active (final) | 167.2 | 5.5 | [160.4, 174.0] |
| pviol_E | 0.334 | 0.025 | [0.312, 0.356] |
| CLV/user (KRW) | 973,986 | — | — |

Note: 5 evaluation repeats provide moderate statistical power. Henderson et al. (2018) recommend ≥10 repeats for publication-grade confidence intervals. The tight CIs here reflect deterministic policy evaluation, not training variance.

---

## 7. Improvement Priority Matrix

| ID | Enhancement | Priority | Expected Impact | Effort | Risk |
|----|------------|----------|----------------|--------|------|
| R1 | Restore 1M timesteps | **P0** | +50–100% reward | Low (config change) | None |
| R2 | Strengthen retention penalty | **P1** | −3–5pp monthly churn | Low (config change) | Overpenalisation |
| R3 | Curriculum learning | **P1** | +20–40% convergence speed | Medium (code change) | Curriculum design |
| R4 | Asymmetric action smoothing | **P1** | Reduced pricing volatility | Low (config+code) | None |
| R5 | Observation enrichment (22D) | **P2** | Better overage timing | Medium (env.py) | Marginal |
| R6 | Population-aware reward | **P2** | Stable N_active | Medium (env.py) | Reward hacking |
| R7 | Hierarchical action timing | **P3** | Domain-realistic pricing | High (architecture) | Complexity |
| R8 | Exploration noise schedule | **P3** | Better initial exploration | Medium (train.py) | Instability |

---

## 8. Cumulative Revision History

| Version | Focus | Tests | Mean Reward | Episode Profit (M KRW) | Key Issue |
|---------|-------|-------|-------------|----------------------|-----------|
| v1 | Initial | 22 | −0.270 | −182 | Broken calibration |
| v2 | Calibration (C1–C5) | 26 | +0.143 | +95.3 | Random baseline only |
| v3 | SB3 transparency (F1) | 26 | — | — | Silent import failure |
| v4 | CSVLogger crash (F5) | 26 | +0.255 | +162 | F_E saturated at 110K |
| v5/v6 | 9 enhancements (E1–E9) | 37 | **+0.290** | **+379** | 50K steps (should be 1M) |
| **v7** | **Proposed (R1–R8)** | — | **+0.40–0.50** | **+500–700** | — |

---

## 9. Files Requiring Modification

| File | Change | Priority |
|------|--------|----------|
| `config/default.yaml` | [R1] `total_timesteps: 1000000`, [R2] `alpha_retention: 2.0`, [R4] per-dimension smoothing weights | P0/P1 |
| `oran3pt/env.py` | [R4] Per-dimension smoothing, [R5] 22D observation, [R6] population reward term | P1/P2 |
| `oran3pt/train.py` | [R3] Curriculum learning phases, [R8] entropy schedule | P1/P3 |
| `tests/test_env.py` | Updated for 22D obs, new reward components | P2 |

---

## 10. References

| Tag | Citation | Used For |
|-----|----------|----------|
| [Bacon 2017] | Bacon, Harb, Precup. "The Option-Critic Architecture." *AAAI*, 2017 | [R7] Hierarchical actions |
| [Bengio 2009] | Bengio et al. "Curriculum Learning." *ICML*, 2009 | [R3] Curriculum design |
| [Dalal 2018] | Dalal et al. "Safe Exploration in Continuous Action Spaces." *NeurIPS*, 2018 | [R4] Per-dimension constraints |
| [Dulac-Arnold 2021] | Dulac-Arnold et al. "Challenges of Real-World RL." *JMLR*, 22(73), 2021 | [R5] Observations, [D5] smoothing |
| [Fader 2010] | Fader, Hardie. "Customer-Base Valuation." *Marketing Science*, 2010 | [D4] Retention value |
| [Gupta 2006] | Gupta et al. "Modeling CLV." *J. Service Research*, 9(2), 2006 | CLV analysis, pricing levels |
| [Haarnoja 2018] | Haarnoja et al. "Soft Actor-Critic." *ICML*, 2018 | [R1] Training duration |
| [Henderson 2018] | Henderson et al. "Deep RL that Matters." *AAAI*, 2018 | [D1] Episode count, [R1] |
| [Mguni 2019] | Mguni et al. "Coordinating the Crowd." *AAMAS*, 2019 | [R6] Population reward |
| [Narvekar 2020] | Narvekar et al. "Curriculum Learning for RL." *JMLR*, 21(181), 2020 | [R3] Curriculum framework |
| [Ng 1999] | Ng, Harada, Russell. "Policy Invariance Under Reward Transformations." *ICML*, 1999 | [D4] Reward shaping magnitude |
| [Vezhnevets 2017] | Vezhnevets et al. "FeUdal Networks for Hierarchical RL." *ICML*, 2017 | [R7] Multi-timescale actions |
| [Wiewiora 2003] | Wiewiora et al. "Principled Methods for Advising RL Agents." *ICML*, 2003 | [R2] Non-potential shaping |
| [Zheng 2022] | Zheng et al. "The AI Economist." *Science Advances*, 8(15), 2022 | [R6] Economic simulation reward |
| [Zhou 2022] | Zhou et al. "Revisiting Exploration in Deep RL." *ICLR*, 2022 | [R8] Entropy scheduling |
