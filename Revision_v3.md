# Revision 4: SAC Training Crash — CSVLogger vs SB3 Episode Boundary

## 1. Root Cause

**Error message:**
```
SAC training failed: dict contains fields not in fieldnames: 'terminal_observation', 'episode'
```

**What happens step by step:**

| Step | Event | Detail |
|------|-------|--------|
| 1 | SAC starts training | Steps 1–345 log correctly to CSV |
| 2 | First episode terminates (step ≈ 346) | `env.step()` returns `terminated=True` |
| 3 | SB3 injects extra keys into `info` dict | `terminal_observation` (numpy array), `episode` (dict with `r`, `l`, `t`) |
| 4 | `_LogCallback._on_step()` passes full `info` to `CSVLogger.log()` | — |
| 5 | `csv.DictWriter.writerow()` sees keys not in `fieldnames` | **Raises `ValueError`** |
| 6 | `except Exception` in `train()` catches it | Falls back to random baseline |
| 7 | 300K steps run with random actions | **Zero learning** |

The `CSVLogger` initialised its `DictWriter` fieldnames from the first step's `info` dict (which only contains environment keys like `step`, `profit`, `revenue`, etc.). When step ~346 terminates the first episode, SB3's `VecEnv` wrapper adds `terminal_observation` and `episode` to the dict. `DictWriter` with default `extrasaction='raise'` throws immediately.

**Why it wasn't caught earlier:** The broad `except Exception` in `train()` silently redirects to the random baseline with an identical-looking tqdm progress bar — making it appear that training continues normally.

## 2. SB3-Injected Keys

SB3 adds these keys to `info` at episode boundaries (see `stable_baselines3/common/vec_env/base_vec_env.py`):

| Key | Type | When Added | Content |
|-----|------|-----------|---------|
| `terminal_observation` | `np.ndarray` | Episode termination | Final observation before reset |
| `episode` | `dict` | Episode termination | `{'r': total_reward, 'l': episode_length, 't': wall_time}` |
| `TimeLimit.truncated` | `bool` | Time-limit truncation | Whether truncation was due to time limit |
| `_final_observation` | `np.ndarray` | Some VecEnv wrappers | Same as terminal_observation |
| `_final_info` | `dict` | Some VecEnv wrappers | Final info before reset |

None of these are scalar values suitable for CSV logging.

## 3. Fix Applied

**Two-layer defense in `oran3pt/train.py`:**

### Layer 1: CSVLogger — `extrasaction='ignore'`

```python
# Before
self._writer = csv.DictWriter(self._f, fieldnames=self._fields)

# After
self._writer = csv.DictWriter(
    self._f, fieldnames=self._fields, extrasaction="ignore"
)
```

This tells `DictWriter` to silently skip any keys not in the original fieldnames, instead of raising `ValueError`.

### Layer 2: _LogCallback — pre-filter non-scalar keys

```python
_SB3_INJECTED_KEYS = frozenset({
    "terminal_observation", "episode",
    "TimeLimit.truncated", "_final_observation", "_final_info",
})

def _filter_info(info: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in info.items() if k not in _SB3_INJECTED_KEYS}

# In _LogCallback._on_step():
self.csvl.log(_filter_info(infos[0]))   # was: self.csvl.log(infos[0])
```

This removes non-scalar SB3 metadata before it reaches the CSV writer. Even without `extrasaction='ignore'`, the filtered dict would only contain environment-produced scalar keys.

### Layer 3: Better error reporting

```python
# Before
logger.error("SAC training failed: %s", e)

# After
logger.error("SAC training failed: %s", e, exc_info=True)
```

Adding `exc_info=True` prints the full traceback, so future errors won't be opaque one-liners.

## 4. Files Modified

| File | Change | Lines |
|------|--------|-------|
| `oran3pt/train.py` | `CSVLogger`: added `extrasaction="ignore"` | ~96 |
| `oran3pt/train.py` | Added `_SB3_INJECTED_KEYS` and `_filter_info()` | ~61–76 |
| `oran3pt/train.py` | `_LogCallback._on_step()`: filter before log | ~163 |
| `oran3pt/train.py` | `except` block: `exc_info=True` for full traceback | ~189 |

No other files changed. `env.py`, `config/default.yaml`, `tests/test_env.py` are unaffected.

## 5. Expected Result After Fix

```
===== Step 3: Train SAC =====
[...] SB3 available — training SAC for 300000 timesteps
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 300,000/300,000  [~1:08:00, 73 it/s]
[...] Model saved → outputs/best_model.zip

===== Step 4: Evaluation =====
Found trained model: outputs/best_model.zip
```

With SAC actually completing 300K steps, expected improvements over random baseline:

| Metric | Random Baseline | Trained (expected) |
|--------|----------------|-------------------|
| Mean reward | +0.143 | +0.3 – +0.5 |
| Episode profit | ~95 M KRW | ~120–180 M KRW |
| Monthly churn | ~6.8% | ~3–4% |
| CLV/user | ~527K KRW | ~700K–1M KRW |