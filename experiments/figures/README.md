# Visualization Results — Presentation Talking Points

The 4 figures in this folder are designed to explain the 5G network slicing
dynamic pricing RL results to an audience at a glance. The one-line summaries
and "say this" sentences next to each figure can be used as-is.

> **Reading order:** `fig_dashboard.png` alone is designed to tell the full story.
> If you have more time, go deeper in order: Figure 2 -> 3 -> 4.

---

## 1. `fig_dashboard.png` — Headline One-page Summary (2x3 panel)

**In one line:** "PPO simultaneously captures revenue and subscriber retention, +162.9% over the static policy."

| Panel | What it shows | Say this |
|---|---|---|
| (a) Revenue vs. Retention | Scatter plot of (total revenue, final eMBB subscribers) for 6 policies | "Upper-right is better. PPO is the only one that captures both axes; Max-Price looks similar in revenue but subscribers have nearly vanished." |
| (b) Net Reward Bar | Average net reward for 6 policies | "PPO achieves 2.6x the net reward of the static policy and +40% over Max-Price." |
| (c) eMBB Subscriber Trend | 4 N_E(t) curves over 720 hours | "Max-Price causes subscribers to plummet within the first 100 hours, while PPO/SAC maintain over 4,000." |
| (d) PPO Price Time Series | Learned F_U(t), F_E(t) | "Prices move over time rather than staying at reference values — this is the 'dynamic' policy." |
| (e) Learning Curves | PPO/SAC training reward trends | "Both algorithms surpass the static policy early in training and converge stably." |
| (f) Eval Distribution | Box plot of 20-episode evaluation results | "The very small standard deviation shows the results are consistent, not coincidental." |

**Note:** The net reward axis uses the environment's internal x10^-5 scaling.
The penalty (~20M USD) is a *structural lower bound* common to all policies, as URLLC QoS is modeled exogenously.

---

## 2. `fig_subscriber_dynamics.png` — Subscriber Dynamics by Policy

**In one line:** "Max-Price's attractive revenue is a side effect of subscriber collapse."

- URLLC slice (left): All policies are relatively stable around 1,000 subscribers.
- eMBB slice (right): Max-Price drops vertically from 5,000 to ~900. PPO/SAC maintain ~4,000 and ~2,200 respectively.
- "**Max-Price maximizes short-term revenue but is unsustainable long-term. PPO learned through training that raising prices too much leads to subscriber loss.**"

---

## 3. `fig_price_trajectories.png` — Pricing Policies Learned by PPO and SAC

**In one line:** "The two algorithms pursue the same reward with different pricing strategies."

- 4 panels: F_U, p_U, F_E, p_E respectively.
- Dashed line = reference price, gray dotted line = action space upper bound.
- Key points for the audience:
  - "You can see PPO and SAC choosing similar prices for some slices and different prices for others."
  - "These are not simply 'raise or lower prices' policies — they learned to respond to slice characteristics and subscriber state."

---

## 4. `fig_revenue_breakdown.png` — Revenue Composition Breakdown

**In one line:** "PPO's advantage comes not from squeezing one slice, but from balancing both."

- (a) Stacked bar: Per-policy URLLC/eMBB revenue + QoS penalty.
- (b) Horizontal bar: eMBB's share of total revenue.
- Key points for the audience:
  - "PPO/SAC's revenue structure is distributed across both slices, making it resilient even if one slice collapses."
  - "Max-Price has fewer subscribers in both slices, shrinking the revenue pool itself."

---

## Brief Answers to Frequently Asked Questions

1. **"Why did PPO outperform SAC?"** — PPO is on-policy, so it adapted faster to the environment where subscriber counts change gradually (non-stationarity).
2. **"Why doesn't the penalty drop to 0?"** — URLLC QoS is exogenous (random variable) in this model, so the 99.999% target is almost always missed. A structural lower bound of ~20M USD is common to all policies. Joint price-resource optimization remains for future work.
3. **"Why does Random have higher reward than Static-Heuristic?"** — Random exploration frequently tries prices above the reference. However, subscriber counts fluctuate unstably, so it is not a sustainable long-term policy.
4. **"Isn't this a single-seed result?"** — Yes, these figures are from seed=42 training + single episode rollout. The evaluation phase supplements reproducibility with a 20-episode distribution (Figure 1f). Multi-seed validation is future work.
