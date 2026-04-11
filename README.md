# Networking-Price

5G 네트워크 슬라이싱(URLLC / eMBB) 환경에서 **수익(Revenue)**과 **QoS 패널티(Penalty)**를 동시에 고려하는 **동적 가격결정(dynamic pricing)** 강화학습 연구 코드.
Stable-Baselines3 의 PPO/SAC 를 사용해 720시간(=1 episode) 길이의 MDP 위에서 정책을 학습하고, 4종 비-RL 베이스라인과 정량 비교한다.

- **상위 디렉토리:** `Networking-Price/` (작업공간)
- **Git 저장소:** `network-pricing/` (이 디렉토리)
- **원격 저장소:** `https://github.com/Ryu-Jemu/network-pricing`

---

## 1. 디렉토리 구조

```
network-pricing/
├── env/                              # Gymnasium 환경 (MDP 정의)
│   ├── __init__.py
│   └── network_slicing_env.py        # NetworkSlicingEnv (5G 슬라이싱 동적 가격)
│
├── train/                            # 학습 / 평가 스크립트
│   ├── __init__.py
│   ├── config.py                     # ENV_CONFIG / SAC_CONFIG / PPO_CONFIG / EVAL_CONFIG
│   ├── train_sac.py                  # SAC 학습 + 20-episode evaluation
│   ├── train_ppo.py                  # PPO 학습 + 20-episode evaluation
│   └── baselines.py                  # Static-Heuristic / Random / Zero-Price / Max-Price 평가
│
├── tests/
│   └── test_env.py                   # 환경 7개 검증 (LogNormal, departure, billing 등)
│
├── experiments/                      # 실험 산출물
│   ├── make_figures.py               # 발표·논문용 4종 그림 생성기
│   ├── models/                       # (gitignored) 학습된 SB3 모델 zip
│   │   ├── ppo_seed42.zip
│   │   └── sac_seed42.zip
│   ├── results/                      # 평가 결과 JSON
│   │   ├── baselines.json
│   │   ├── ppo_seed42.json
│   │   └── sac_seed42.json
│   └── figures/                      # 시각화 결과 PNG (4종) + 발표 가이드 README
│       ├── README.md
│       ├── fig_dashboard.png
│       ├── fig_subscriber_dynamics.png
│       ├── fig_price_trajectories.png
│       └── fig_revenue_breakdown.png
│
├── paper/                            # (gitignored) LaTeX 논문 초안
│   ├── draft.tex
│   └── main.tex
├── Example/                          # (gitignored) IEEEtran LaTeX 템플릿
├── Reference/                        # (gitignored) 참고 논문 PDF
├── Architecture (1).svg              # (gitignored) 시스템 아키텍처 다이어그램
├── Time step (1).svg                 # (gitignored) 단일 time step 다이어그램
├── Model Spec (1).pdf                # (gitignored) 환경 모델 명세서 (F1-F11)
├── model-architecture.md             # (gitignored) 모델 아키텍처 문서
└── .gitignore
```

> **참고:** `paper/`, `Example/`, `Reference/`, `*.svg`, `*.pdf`, `*.doc`, `*.zip`, `model-architecture.md` 등은 디스크에는 존재하지만 `.gitignore` 에 의해 추적 제외되어 있다. `.gitignore` 의 정확한 규칙은 [.gitignore](.gitignore) 를 참조.

---

## 2. 환경: `NetworkSlicingEnv`

[`env/network_slicing_env.py`](env/network_slicing_env.py) 는 Gymnasium API 를 따르는 720-step MDP 다.
환경 명세는 `Model Spec (1).pdf` 의 F1–F11 공식과 1:1 대응한다.

| 항목 | 내용 |
|---|---|
| 슬라이스 | URLLC (s=0), eMBB (s=1) |
| 상태 | (N_U, N_E, η_U_prev, η_E_prev) — 4-dim, 모두 정규화 |
| 행동 | (F_U, p_U, F_E, p_E) — `Box([0,1]^4)`, step 내부에서 실제 가격으로 스케일 |
| 보상 | `r_t = (Revenue − QoS Penalty) × reward_scale` (`reward_scale = 1e-5`) |
| Episode 길이 | T = 720 (1시간 단위, 30일에 해당) |
| 할인율 γ | 0.99 |

각 step 은 4개 phase 로 구성된다:

1. **Phase B — 가입자 동역학**
   - F3 Departure: `P_dep = σ(γ0 + γ_F·F̃ + γ_p·p̃ − γ_η·η_prev)` → Binomial 추첨
   - F4/F5 Arrival: `P_arr = σ(β0 − β_F·F̃ − β_p·p̃)`, `λ = λ_max·P_arr` → Poisson 추첨
2. **Phase C — 사용량/QoS**
   - F1/F2 Billing: 사용량 q ~ LogNormal(μ_s, σ_s²), 청구액 `B = F_s + max(0, q − Q̄_s)·p_s`
   - F7 QoS: η_s ~ Uniform(η_low, η_high) (외생 모델)
3. **Phase D — 보상**
   - F8/F9 Penalty: `Σ_s w_s · N_active_s · max(0, η_tgt_s − η_s)`
   - F10: `r = Revenue − Penalty`, scale 적용
4. **Phase A — 상태 전이** (F11)

### 주요 환경 파라미터 (Audit-검증)

[`train/config.py`](train/config.py) `ENV_CONFIG` 참조.

| 파라미터 | URLLC | eMBB |
|---|---|---|
| 기준 정액료 `F_ref` | 50 | 30 |
| 기준 종량료 `p_ref` | 10 | 5 |
| 무료 할당량 `Q̄` | 5 | 30 |
| 행동 상한 `F_max / p_max` | 100 / 20 | 100 / 20 |
| 트래픽 LogNormal `(μ, σ²)` | (1.0, 0.5) | (3.0, 0.8) |
| Departure `γ0` | −9.72 | −12.12 |
| `γ_F / γ_p / γ_η` | 1.0 / 0.8 / 3.0 | 1.0 / 0.8 / 0.5 |
| Arrival `β0` | 2.0 | 2.5 |
| Arrival `λ_max` | 0.05 | 0.15 |
| QoS `η_tgt` | 0.99999 | 0.90 |
| Penalty 가중치 `w` | 500 | 50 |
| 초기 가입자 `N_init` | 1,000 | 5,000 |

---

## 3. 알고리즘 / 학습 설정

[`train/config.py`](train/config.py) 참조. 모두 `seed=42` 단일 시드 학습.

| 알고리즘 | 학습 episode | 정책 네트워크 | 주요 하이퍼파라미터 |
|---|---|---|---|
| **SAC** ([`train/train_sac.py`](train/train_sac.py)) | 500 | `[256, 256, 256]` MLP | lr=3e-4, batch=256, buffer=1M, τ=0.005, ent_coef=auto |
| **PPO** ([`train/train_ppo.py`](train/train_ppo.py)) | 1,000 | `[256, 256, 256]` MLP | lr=3e-4, batch=64, n_epochs=10, clip=0.2, GAE λ=0.95 |

평가는 두 알고리즘 모두 학습 후 deterministic policy 로 **20 episode** 를 돌려 mean ± std 를 보고한다.

### 베이스라인 ([`train/baselines.py`](train/baselines.py))

| 베이스라인 | 정책 |
|---|---|
| Static-Heuristic | 항상 `[0.5, 0.5, 0.3, 0.25]` (= 기준 정액/종량료) 적용 |
| Random | 매 step `Uniform(0,1)^4` |
| Zero-Price | `[0, 0, 0, 0]` (수익 0) |
| Max-Price | `[1, 1, 1, 1]` (`F=100, p=20` 양 슬라이스) |

---

## 4. 결과

> 모든 수치는 [`experiments/results/`](experiments/results/) 의 JSON 파일에서 직접 산출한 것이다.
> Reward / Revenue / Penalty 는 모두 **20-episode evaluation 평균**.
> Reward 는 환경 내부에서 `×1e-5` 스케일링 된 값이다.

### 4-1. 정량 비교 (mean over 20 eval episodes)

| 정책 | Reward (scaled) ↑ | Revenue (USD) | Penalty (USD) | 최종 N_U | 최종 N_E |
|---|---:|---:|---:|---:|---:|
| **PPO (제안)**     | **8,194 ± 38** | **840,809,875** | 21,412,713 | 950 | **4,102** |
| **SAC (제안)**     | 7,176 ± 64     | 738,471,348     | 20,838,421 | 990 | 2,188      |
| Max-Price          | 5,870 ± 48     | 606,322,112     | 19,300,358 | 921 | 909        |
| Random             | 5,342 ± 48     | 556,418,152     | 22,238,233 | 1,003 | 4,358    |
| Static-Heuristic   | 3,117 ± 5      | 334,320,303     | 22,597,244 | 1,006 | 5,003    |
| Zero-Price         | −228 ± 4       | 0               | 22,802,477 | 1,028 | 5,090    |

### 4-2. 핵심 관찰

- **PPO 가 모든 지표에서 우위.** Reward 기준으로 Static-Heuristic 대비 **+162.9%**, Max-Price 대비 **+39.6%**.
- **Max-Price 의 함정.** Revenue 만 보면 Max-Price 가 PPO 다음이지만, 최종 eMBB 가입자가 5,000 → **909** 로 붕괴해 장기적으로 지속 불가능하다. PPO 는 수익을 극대화하면서도 eMBB 가입자 4,100명을 유지한다.
- **QoS 패널티의 구조적 하한.** URLLC QoS 가 외생 random variable 로 모델링되어 99.999% 목표가 거의 항상 미달된다. 약 **20–22M USD** 의 패널티는 모든 정책에 공통으로 깔리는 *하한* 이다 (Zero-Price 도 22.8M).
- **학습 곡선.** PPO 는 1,000 episode 동안 reward 3,236 → 7,729 (마지막 100 ep 평균), SAC 는 500 episode 동안 6,807 → 7,268 로 수렴.

### 4-3. 시각화 — `experiments/figures/`

[`experiments/make_figures.py`](experiments/make_figures.py) 가 4종 그림을 생성한다. 각 그림에 대한 발표 가이드는 [`experiments/figures/README.md`](experiments/figures/README.md) 참조.

| 파일 | 내용 | 한 줄 요약 |
|---|---|---|
| [`fig_dashboard.png`](experiments/figures/fig_dashboard.png) | 2×3 헤드라인 패널 (수익-유지 산점도, 보상 막대, 가입자 추이, 가격 시계열, 학습 곡선, eval 분포) | 한 장에 모든 결과 |
| [`fig_subscriber_dynamics.png`](experiments/figures/fig_subscriber_dynamics.png) | 720시간 동안 N_U(t), N_E(t) 정책별 비교 | Max-Price 의 가입자 붕괴 가시화 |
| [`fig_price_trajectories.png`](experiments/figures/fig_price_trajectories.png) | PPO/SAC 가 학습한 F_U, p_U, F_E, p_E 시계열 | 정적 가격이 아닌 *동적* 정책 |
| [`fig_revenue_breakdown.png`](experiments/figures/fig_revenue_breakdown.png) | 정책별 URLLC/eMBB 수익 적층 + eMBB 비중 | PPO 의 우위는 양 슬라이스 균형에서 |

---

## 5. 재현 절차

### 5-1. 의존성

`requirements.txt` 는 현재 저장소에 포함되어 있지 않다. 코드가 import 하는 핵심 패키지:

```
gymnasium
numpy
scipy
matplotlib
stable-baselines3
torch        # SB3 백엔드
tqdm         # SB3 progress bar
```

수동 설치 예시:
```bash
pip install gymnasium numpy scipy matplotlib stable-baselines3[extra]
```

### 5-2. 환경 검증 (≈30초)

```bash
cd /Users/ryujemu/Desktop/LLM_Resource_Allocation/Networking-Price/network-pricing
python3 tests/test_env.py
```

7개 단위 검증 통과 시 `ALL TESTS PASSED` 가 출력된다. 검증 항목은
LogNormal `E[q]` / `P(q>Q̄)`, Departure 확률 `σ(−10.77)=2.10e-5`,
월간 churn 1.502%, Arrival 확률, 기대 청구액 `E[B_U]=$55.59 / E[B_E]=$81.67`,
환경 step/full episode/edge case (Zero-Price → revenue=0).

### 5-3. 베이스라인 평가 (≈1분)

```bash
python3 train/baselines.py
# → experiments/results/baselines.json
```

### 5-4. SAC / PPO 학습 + 평가

```bash
python3 train/train_sac.py
# → experiments/models/sac_seed42.zip
# → experiments/results/sac_seed42.json   (500 episodes 학습 + 20 eval)

python3 train/train_ppo.py
# → experiments/models/ppo_seed42.zip
# → experiments/results/ppo_seed42.json   (1000 episodes 학습 + 20 eval)
```

> SAC 는 약 360k step (500 ep × 720), PPO 는 720k step (1000 ep × 720) 을 돈다.
> CPU 만으로 가능하지만 GPU 가 있으면 더 빠르다.

### 5-5. 그림 생성

```bash
python3 experiments/make_figures.py
# → experiments/figures/fig_*.png (4개)
```

PPO/SAC 학습 결과(`experiments/models/*.zip`, `experiments/results/*.json`) 가 모두 존재해야 동작한다.

---

## 6. 한계 / 향후 작업

- **단일 시드 (seed=42).** 본 결과는 단일 시드 학습 + 20-episode evaluation 분포다. `EVAL_CONFIG["seeds"] = [42, 123, 456, 789, 1024]` 가 정의되어 있으나 다중 시드 실행은 아직 미진행.
- **외생적 QoS.** η 가 Uniform 분포로 외생 모델링되어, URLLC 99.999% 목표가 정책으로 개선되지 않는다. 향후 가격–자원할당 결합 최적화로 확장 가능.
- **2개 슬라이스 한정.** URLLC / eMBB 만 다루며 mMTC 등은 미포함.
- **고정 사용자 행동 모델.** 가입자 의사결정이 sigmoid logit 으로 단순화되어 있어 실제 시장 데이터로의 보정이 필요.

---

## 7. 라이선스 / 인용

학술 발표용 코드 (KIPS 학술발표대회 제출). 라이선스는 별도 명시 전이며 인용 시 저장소 URL 사용.
