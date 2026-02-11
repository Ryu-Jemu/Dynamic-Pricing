# Revision 3: Root Cause Analysis & Learning Efficiency Fixes

## 1. Executive Summary

The pipeline runs to completion but **no learning occurs**. All 300,000 training steps execute random actions because `stable-baselines3` fails to import silently. The entire pipeline output (training logs, evaluation, CLV, dashboard) reflects only a **random baseline** — not a trained agent.

---

## 2. Root Cause Analysis

### 2.1 [CRITICAL] SB3 Import Failure — Zero Learning

**Symptom:**
```
[oran3pt.train] SB3 not installed — running random baseline.
```
despite `stable-baselines3>=2.3.0` being listed in `requirements.txt`.

**Root Cause Chain:**

| Step | What Happens | Impact |
|------|-------------|--------|
| 1 | `run.sh` installs dependencies with `pip install -r requirements.txt -q 2>&1 \| tail -3` | **Installation errors are suppressed** — only last 3 lines of output visible |
| 2 | SB3 install fails silently (likely dependency conflict with torch/gymnasium versions) | No error shown to user |
| 3 | `train.py` wraps `from stable_baselines3 import SAC` in `try/except ImportError` | **Failure is caught and swallowed** |
| 4 | Fallback to `_run_random_baseline()` with identical tqdm bar | User sees progress bar, assumes training is occurring |
| 5 | No `best_model.zip` is saved | — |
| 6 | `run.sh` checks `if [ -f "outputs/best_model.zip" ]` — file doesn't exist | Evaluation also uses random policy |
| 7 | Dashboard, CLV, summary all reflect random-action performance | **All outputs are meaningless for model maturity** |

**Evidence:** PyTorch imports successfully (`PyTorch device: mps`), confirming the issue is specific to SB3, not the Python environment.

**Probable SB3 failure modes on macOS/Apple Silicon:**
1. `shimmy` version conflict with gymnasium 0.29+
2. `ale-py` (Atari) C extension build failure on ARM64
3. Version pinning conflict between `torch>=2.0.0` and SB3's internal torch requirements

### 2.2 [MODERATE] Pandas FutureWarning in CLV

**Symptom:**
```
FutureWarning: DataFrameGroupBy.apply operated on the grouping columns.
```

**Root Cause:** `eval.py` line 107 uses `df.groupby("repeat").apply(lambda g: ...)` without `include_groups=False`. In pandas ≥2.2, groupby columns are included in the lambda input, and this behavior will change in a future version.

**Impact:** Will become a hard error in pandas 3.0, breaking CLV reporting.

### 2.3 [MODERATE] Streamlit API Deprecation

**Symptom:**
```
Please replace `use_container_width` with `width`.
`use_container_width` will be removed after 2025-12-31.
```

**Root Cause:** `dashboard_app.py` uses `st.plotly_chart(fig, use_container_width=True)` across 10 chart calls. The Streamlit API changed this parameter to `width="stretch"`.

**Impact:** Dashboard will break after Streamlit drops the deprecated parameter.

---

## 3. Fixes Applied

### Fix F1: Transparent SB3 Installation & Import Verification

**File: `scripts/run.sh`**

| Before | After |
|--------|-------|
| `pip install -r requirements.txt -q 2>&1 \| tail -3` | `pip install -r requirements.txt 2>&1 \| tee /tmp/pip_install.log \| tail -5` |
| No import verification | Python verification block checks SB3, torch, gymnasium |
| No fallback install | Targeted `pip install "stable-baselines3[extra]"` if initial install fails |

**File: `oran3pt/train.py`**

| Before | After |
|--------|-------|
| `try: from sb3 import SAC except ImportError: ...` (buried in `train()`) | Module-level import with `_SB3_AVAILABLE` flag and `_SB3_IMPORT_ERROR` message |
| Silent `logger.warning("SB3 not installed")` | Explicit `logger.error()` with install instructions and visual separator |

**File: `requirements.txt`**

| Before | After |
|--------|-------|
| `stable-baselines3>=2.3.0` | `stable-baselines3[extra]>=2.3.0` |

The `[extra]` variant ensures all optional SB3 dependencies (tensorboard, shimmy, etc.) are installed, reducing the chance of partial-install failures.

### Fix F3: Pandas FutureWarning

**File: `oran3pt/eval.py`** line 107

```python
# Before
monthly_profit = df.groupby("repeat").apply(
    lambda g: g.groupby(...)["profit"].sum().mean()
)

# After
monthly_profit = df.groupby("repeat").apply(
    lambda g: g.groupby(...)["profit"].sum().mean(),
    include_groups=False,
)
```

### Fix F4: Streamlit Deprecation

**File: `oran3pt/dashboard_app.py`** — all 10 occurrences

```python
# Before
st.plotly_chart(fig, use_container_width=True)

# After
st.plotly_chart(fig, width="stretch")
```

---

## 4. Impact Assessment

### Before Fixes (Current State)

| Aspect | Status | Problem |
|--------|--------|---------|
| SB3 Training | ❌ Not running | ImportError silently caught |
| Model file | ❌ Not created | No `best_model.zip` |
| Evaluation | ⚠️ Random only | No trained policy to evaluate |
| Dashboard | ⚠️ Deprecated API | Will break in future Streamlit |
| CLV report | ⚠️ FutureWarning | Will break in future pandas |
| **Model maturity** | **None** | **Equivalent to untrained random policy** |

### After Fixes (Expected)

| Aspect | Status | Improvement |
|--------|--------|-------------|
| SB3 Training | ✅ SAC trains 300K steps | Agent learns pricing policy |
| Model file | ✅ `best_model.zip` saved | Persistent trained model |
| Evaluation | ✅ Trained policy evaluated | Meaningful performance metrics |
| Dashboard | ✅ No deprecation warnings | Forward-compatible |
| CLV report | ✅ No FutureWarning | Forward-compatible |
| **Model maturity** | **Trained** | **Reward improvement over random baseline** |

### Expected Performance Improvement (Based on Revision 2 Analysis)

With SAC actually training, the Revision 2 calibration results suggest:

| Metric | Random Baseline | Expected Trained | Source |
|--------|----------------|-----------------|--------|
| Mean reward | +0.143 | +0.3 – +0.5 | SAC converges above random |
| Episode profit | ~95 M KRW | ~120–180 M KRW | Learned pricing optimization |
| Monthly churn | ~6.8% (stochastic) | ~3–4% | Agent learns to avoid over-pricing |
| pviol_E | ~0.66 | ~0.3–0.5 | Agent learns ρ_U allocation |
| CLV/user | ~527K KRW | ~700K–1M KRW | Higher retention + revenue |

---

## 5. Files Modified

| File | Changes | Category |
|------|---------|----------|
| `scripts/run.sh` | [F1] Transparent pip, import verification, fallback install | Critical |
| `oran3pt/train.py` | [F1] Module-level SB3 check, explicit error reporting | Critical |
| `requirements.txt` | [F1] `stable-baselines3[extra]` | Critical |
| `oran3pt/eval.py` | [F3] `include_groups=False` in groupby.apply | Moderate |
| `oran3pt/dashboard_app.py` | [F4] `width="stretch"` replacing deprecated param | Moderate |

Files unchanged: `config/default.yaml`, `oran3pt/env.py`, `oran3pt/utils.py`, `oran3pt/gen_users.py`, `tests/test_env.py`.

---

## 6. Verification Steps

After applying fixes, re-run the pipeline and verify:

```bash
# 1. SB3 imports successfully
python -c "from stable_baselines3 import SAC; print('OK')"

# 2. Training produces a model file
ls -la outputs/best_model.zip

# 3. No warnings in evaluation
python -m oran3pt.eval --config config/default.yaml 2>&1 | grep -i "warning\|error"

# 4. No deprecation in dashboard
streamlit run oran3pt/dashboard_app.py 2>&1 | grep -i "deprecated"

# 5. Trained agent outperforms random baseline
python -c "
import pandas as pd
df = pd.read_csv('outputs/rollout_log.csv')
print(f'Mean reward: {df[\"reward\"].mean():.4f}')
print(f'Mean profit: {df[\"profit\"].mean():.0f} KRW')
"
```

---

## 7. Recommendations for Further Model Maturity

| Priority | Enhancement | Rationale |
|----------|------------|-----------|
| P0 | **Resolve SB3 install** | Without this, no learning occurs |
| P1 | **Multi-seed training** (≥5 seeds) | Henderson et al. (AAAI 2018) — RL results require statistical confidence |
| P1 | **EvalCallback** with best-model checkpointing | Save model at peak eval performance, not end of training |
| P2 | **Increase timesteps to 500K–1M** | 300K may be insufficient for 5D continuous action SAC |
| P2 | **Learning rate schedule** | Cosine decay from 3e-4 to 1e-5 improves late-stage convergence |
| P3 | **Curriculum learning** | Start with expectation mode, switch to stochastic after convergence |