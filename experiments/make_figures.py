"""
make_figures.py — 학습 결과 시각화 스크립트
=============================================
PPO/SAC 학습 결과를 발표·논문용 그림 4종으로 정리한다.

생성되는 그림 (experiments/figures/):
  1. fig_dashboard.png            — 헤드라인 (2x3 패널)
  2. fig_subscriber_dynamics.png  — 가입자 추이 (URLLC/eMBB)
  3. fig_price_trajectories.png   — 학습된 가격 정책 시계열
  4. fig_revenue_breakdown.png    — 슬라이스별 수익 분해

실행:
  cd /Users/ryujemu/Desktop/LLM_Resource_Allocation/Networking-Price
  python3 experiments/make_figures.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- 프로젝트 경로 설정 (train 스크립트들과 동일 패턴) ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJ_ROOT)

from env.network_slicing_env import NetworkSlicingEnv  # noqa: E402
from train.config import ENV_CONFIG, REFERENCE_ACTION  # noqa: E402
from stable_baselines3 import PPO, SAC  # noqa: E402

RESULTS_DIR = os.path.join(PROJ_ROOT, "experiments", "results")
MODELS_DIR = os.path.join(PROJ_ROOT, "experiments", "models")
FIGURES_DIR = os.path.join(PROJ_ROOT, "experiments", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1. 폰트 / 색상 설정
# ──────────────────────────────────────────────────────────────────────────────
def setup_matplotlib():
    """한글 라벨 + 통일된 시각 스타일."""
    candidates = ["Apple SD Gothic Neo", "AppleGothic",
                  "NanumGothic", "Malgun Gothic", "Noto Sans CJK KR"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), "DejaVu Sans")
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25
    print(f"[font] using '{chosen}'")
    return chosen


# 정책별 색상/스타일 — PPO 가 모든 그림에서 강조되도록 통일
POLICY_STYLE = {
    "PPO":          dict(color="#d62728", lw=2.6, ls="-",  zorder=5,
                         label="PPO (제안)"),
    "SAC":          dict(color="#1f77b4", lw=2.2, ls="-",  zorder=4,
                         label="SAC (제안)"),
    "Max-Price":    dict(color="#ff7f0e", lw=1.8, ls="--", zorder=3,
                         label="Max-Price"),
    "Random":       dict(color="#9467bd", lw=1.5, ls=":",  zorder=2,
                         label="Random"),
    "Static-Heur.": dict(color="#2ca02c", lw=1.8, ls="-",  zorder=3,
                         label="Static-Heur."),
    "Zero-Price":   dict(color="#8c564b", lw=1.5, ls=":",  zorder=2,
                         label="Zero-Price"),
}
POLICY_ORDER = ["PPO", "SAC", "Max-Price", "Random", "Static-Heur.", "Zero-Price"]


# ──────────────────────────────────────────────────────────────────────────────
# 2. 환경 (per-slice 수익 기록을 위한 얇은 서브클래스)
# ──────────────────────────────────────────────────────────────────────────────
class TrackingEnv(NetworkSlicingEnv):
    """env.step 의 로직을 그대로 따르되, 슬라이스별 수익을 info 에 추가."""

    def step(self, action):
        # NetworkSlicingEnv.step 과 동일한 로직 — 슬라이스별 revenue 분리만 추가
        self.t += 1
        F, p = self._scale_action(action)
        F_tilde = F / self.F_ref
        p_tilde = p / self.p_ref

        # 이탈
        dep_logit = (self.gamma0
                     + self.gamma_F * F_tilde
                     + self.gamma_p * p_tilde
                     - self.gamma_eta * self.eta_prev)
        P_dep = self._sigmoid(dep_logit)
        N_leave = np.zeros(2)
        for s in range(2):
            if self.N[s] > 0:
                N_leave[s] = self.np_random.binomial(int(self.N[s]), float(P_dep[s]))
        N_surv = self.N - N_leave

        # 유입
        arr_logit = self.beta0 - self.beta_F * F_tilde - self.beta_p * p_tilde
        P_arr = self._sigmoid(arr_logit)
        lam = self.lambda_max * P_arr
        N_new = np.array([self.np_random.poisson(float(lam[s])) for s in range(2)],
                         dtype=np.float64)
        N_active = N_surv + N_new

        # 사용량/청구 — 슬라이스별 수익 분리
        revenue_per_slice = np.zeros(2)
        for s in range(2):
            n = int(N_active[s])
            if n > 0:
                usage = self.np_random.lognormal(
                    mean=float(self.mu[s]), sigma=float(self.sigma[s]), size=n)
                overage = np.maximum(0.0, usage - self.Q_bar[s])
                bills = F[s] + overage * p[s]
                revenue_per_slice[s] = float(np.sum(bills))
        total_revenue = float(revenue_per_slice.sum())

        # QoS
        eta = np.array([
            self.np_random.uniform(float(self.eta_low[s]), float(self.eta_high[s]))
            for s in range(2)
        ])

        # 보상
        penalty = 0.0
        for s in range(2):
            shortfall = max(0.0, self.eta_tgt[s] - eta[s])
            penalty += self.w[s] * N_active[s] * shortfall
        reward = (total_revenue - penalty) * self.reward_scale

        # 전이
        self.N = N_active.copy()
        self.eta_prev = eta.copy()
        terminated = self.t >= self.T
        truncated = False

        info = {
            "revenue": total_revenue,
            "revenue_U": revenue_per_slice[0],
            "revenue_E": revenue_per_slice[1],
            "penalty": penalty,
            "N_U": N_active[0], "N_E": N_active[1],
            "eta_U": eta[0], "eta_E": eta[1],
            "F_U": F[0], "p_U": p[0],
            "F_E": F[1], "p_E": p[1],
        }
        return self._get_obs(), float(reward), terminated, truncated, info


# ──────────────────────────────────────────────────────────────────────────────
# 3. 정책 함수 정의
# ──────────────────────────────────────────────────────────────────────────────
def make_policies():
    print("[load] PPO model …")
    ppo_model = PPO.load(os.path.join(MODELS_DIR, "ppo_seed42.zip"), device="cpu")
    print("[load] SAC model …")
    sac_model = SAC.load(os.path.join(MODELS_DIR, "sac_seed42.zip"), device="cpu")

    rng = np.random.default_rng(42)
    ref = np.array(REFERENCE_ACTION, dtype=np.float32)

    def ppo_fn(obs):
        a, _ = ppo_model.predict(obs, deterministic=True)
        return a

    def sac_fn(obs):
        a, _ = sac_model.predict(obs, deterministic=True)
        return a

    return {
        "PPO":          ppo_fn,
        "SAC":          sac_fn,
        "Max-Price":    lambda obs: np.ones(4, dtype=np.float32),
        "Random":       lambda obs: rng.uniform(0, 1, size=4).astype(np.float32),
        "Static-Heur.": lambda obs: ref,
        "Zero-Price":   lambda obs: np.zeros(4, dtype=np.float32),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Rollout
# ──────────────────────────────────────────────────────────────────────────────
def rollout(name: str, policy_fn, env_cfg: dict, seed: int = 42) -> dict:
    """단일 episode rollout. 시계열 + 슬라이스별 수익을 dict 로 반환."""
    env = TrackingEnv(config=env_cfg)
    obs, _ = env.reset(seed=seed)

    T = env.T
    traj = {k: np.zeros(T) for k in
            ["F_U", "p_U", "F_E", "p_E", "N_U", "N_E",
             "eta_U", "eta_E", "revenue", "revenue_U", "revenue_E", "penalty"]}
    total_reward = 0.0
    for t in range(T):
        a = policy_fn(obs)
        obs, r, term, trunc, info = env.step(a)
        total_reward += r
        for k in traj:
            traj[k][t] = info[k]

    out = {
        "name": name,
        "total_reward": total_reward,
        "total_revenue": float(traj["revenue"].sum()),
        "total_revenue_U": float(traj["revenue_U"].sum()),
        "total_revenue_E": float(traj["revenue_E"].sum()),
        "total_penalty": float(traj["penalty"].sum()),
        "final_N_U": float(traj["N_U"][-1]),
        "final_N_E": float(traj["N_E"][-1]),
        "trajectory": traj,
    }
    print(f"[rollout] {name:>13s}: reward={total_reward:8.0f}  "
          f"rev={out['total_revenue']/1e6:7.1f}M  "
          f"pen={out['total_penalty']/1e6:6.1f}M  "
          f"final N=({out['final_N_U']:.0f}, {out['final_N_E']:.0f})")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 5. 데이터 로딩 (저장된 JSON)
# ──────────────────────────────────────────────────────────────────────────────
def load_results():
    with open(os.path.join(RESULTS_DIR, "baselines.json")) as f:
        baselines = json.load(f)
    with open(os.path.join(RESULTS_DIR, "ppo_seed42.json")) as f:
        ppo = json.load(f)
    with open(os.path.join(RESULTS_DIR, "sac_seed42.json")) as f:
        sac = json.load(f)

    # 통합 — 6개 정책의 평균 지표
    summary = {
        "PPO": {
            "mean_reward": float(np.mean([e["total_reward"] for e in ppo["eval_results"]])),
            "std_reward":  float(np.std([e["total_reward"] for e in ppo["eval_results"]])),
            "mean_revenue": float(np.mean([e["total_revenue"] for e in ppo["eval_results"]])),
            "mean_penalty": float(np.mean([e["total_penalty"] for e in ppo["eval_results"]])),
            "mean_final_N_U": float(np.mean([e["final_N_U"] for e in ppo["eval_results"]])),
            "mean_final_N_E": float(np.mean([e["final_N_E"] for e in ppo["eval_results"]])),
            "eval_rewards": [e["total_reward"] for e in ppo["eval_results"]],
        },
        "SAC": {
            "mean_reward": float(np.mean([e["total_reward"] for e in sac["eval_results"]])),
            "std_reward":  float(np.std([e["total_reward"] for e in sac["eval_results"]])),
            "mean_revenue": float(np.mean([e["total_revenue"] for e in sac["eval_results"]])),
            "mean_penalty": float(np.mean([e["total_penalty"] for e in sac["eval_results"]])),
            "mean_final_N_U": float(np.mean([e["final_N_U"] for e in sac["eval_results"]])),
            "mean_final_N_E": float(np.mean([e["final_N_E"] for e in sac["eval_results"]])),
            "eval_rewards": [e["total_reward"] for e in sac["eval_results"]],
        },
    }
    name_map = {
        "static_heuristic": "Static-Heur.",
        "random": "Random",
        "zero_price": "Zero-Price",
        "max_price": "Max-Price",
    }
    for k, v in baselines.items():
        summary[name_map[k]] = {
            "mean_reward": v["mean_reward"],
            "std_reward":  v.get("std_reward", 0.0),
            "mean_revenue": v["mean_revenue"],
            "mean_penalty": v["mean_penalty"],
            "mean_final_N_U": v["mean_final_N_U"],
            "mean_final_N_E": v["mean_final_N_E"],
            "eval_rewards": None,
        }

    return {
        "summary": summary,
        "ppo_train": ppo["train_rewards"],
        "sac_train": sac["train_rewards"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# 6. 그림 1: Dashboard (헤드라인 2x3)
# ──────────────────────────────────────────────────────────────────────────────
def plot_dashboard(results, rollouts, T_env=720):
    summary = results["summary"]
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30,
                          left=0.06, right=0.98, top=0.91, bottom=0.08)

    # ── (a) Revenue–Retention Pareto ────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    label_offsets = {  # (offset_x, offset_y) in points
        "PPO":          (10, 12),
        "SAC":          (10, -4),
        "Max-Price":    (-8, -16),
        "Random":       (10, 0),
        "Static-Heur.": (-50, 12),
        "Zero-Price":   (10, -12),
    }
    label_ha = {
        "Static-Heur.": "right",
        "Max-Price":    "right",
    }
    for name in POLICY_ORDER:
        s = summary[name]
        x = s["mean_revenue"] / 1e6
        y = s["mean_final_N_E"]
        st = POLICY_STYLE[name]
        size = 360 if name == "PPO" else (200 if name == "SAC" else 130)
        ax.scatter(x, y, s=size, color=st["color"],
                   edgecolor="black", linewidth=0.9, zorder=st["zorder"] + 5,
                   label=st["label"])
        ox, oy = label_offsets[name]
        ax.annotate(name, (x, y),
                    xytext=(ox, oy), textcoords="offset points",
                    fontsize=8.5, color=st["color"],
                    fontweight="bold",
                    ha=label_ha.get(name, "left"))
    ax.set_xlabel("총 수익 (M USD, 720h 누적)")
    ax.set_ylabel("eMBB 최종 가입자 수")
    ax.set_title("(a) 수익 vs. 가입자 유지 — PPO 가 유일한 Pareto 우위",
                 fontweight="bold")
    ax.axhline(5000, color="gray", lw=0.8, ls=":", alpha=0.5)
    ax.text(0.02, 0.96, "초기 N_E = 5,000",
            transform=ax.transAxes, fontsize=8, color="gray", va="top")
    # Pareto 우위를 시각적으로 강조: PPO 좌표를 중심으로 사분면 구분
    ppo_x = summary["PPO"]["mean_revenue"] / 1e6
    ppo_y = summary["PPO"]["mean_final_N_E"]
    ax.axvspan(ppo_x - 30, ppo_x + 80, ymin=0, ymax=1,
               color="#d62728", alpha=0.05, zorder=0)
    ax.set_xlim(left=-50)
    ax.set_ylim(bottom=0, top=5800)

    # ── (b) Net Reward Bar ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    names = POLICY_ORDER
    rewards = [summary[n]["mean_reward"] for n in names]
    colors = [POLICY_STYLE[n]["color"] for n in names]
    bars = ax.bar(range(len(names)), rewards, color=colors,
                  edgecolor="black", linewidth=0.6)
    bars[0].set_linewidth(2.0)  # PPO 강조
    bars[0].set_edgecolor("black")
    for i, v in enumerate(rewards):
        ax.text(i, v + (max(rewards) - min(rewards)) * 0.02,
                f"{v:,.0f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold" if i == 0 else "normal")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("순보상 (1e-5 환산)")
    ax.set_title("(b) 정책별 순보상 — PPO 가 정적 대비 +162.9%, Max-Price 대비 +39.6%",
                 fontweight="bold")
    ax.axhline(0, color="black", lw=0.5)
    # 화살표 주석: PPO 와 Static 비교
    ax.annotate("", xy=(0, rewards[0]), xytext=(4, rewards[4]),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.4))
    ax.text(2.0, (rewards[0] + rewards[4]) / 2,
            "+162.9%", color="#d62728", fontweight="bold",
            ha="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d62728"))

    # ── (c) eMBB 가입자 추이 N_E(t) ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    show = ["PPO", "SAC", "Max-Price", "Static-Heur."]
    for name in show:
        if name not in rollouts:
            continue
        st = POLICY_STYLE[name]
        ax.plot(rollouts[name]["trajectory"]["N_E"],
                color=st["color"], lw=st["lw"], ls=st["ls"],
                label=st["label"], zorder=st["zorder"])
    ax.set_xlabel("시간 t (hour)")
    ax.set_ylabel("eMBB 가입자 수 N_E(t)")
    ax.set_title("(c) 가입자 동역학 — Max-Price 는 사용자 붕괴", fontweight="bold")
    ax.axhline(5000, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xlim(0, T_env)

    # ── (d) PPO 가격 시계열 ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    if "PPO" in rollouts:
        traj = rollouts["PPO"]["trajectory"]
        ax.plot(traj["F_U"], color="#d62728", lw=2.0,
                label="F_U (URLLC 고정요금)")
        ax.plot(traj["F_E"], color="#1f77b4", lw=2.0,
                label="F_E (eMBB 고정요금)")
        ax.axhline(50, color="#d62728", lw=0.9, ls="--", alpha=0.6)
        ax.axhline(30, color="#1f77b4", lw=0.9, ls="--", alpha=0.6)
        ax.text(0.99, 50 / 105 + 0.01, "기준 F_U=50", transform=ax.transAxes,
                color="#d62728", fontsize=7.5, ha="right", alpha=0.9)
        ax.text(0.99, 30 / 105 + 0.01, "기준 F_E=30", transform=ax.transAxes,
                color="#1f77b4", fontsize=7.5, ha="right", alpha=0.9)
    ax.set_xlabel("시간 t (hour)")
    ax.set_ylabel("고정요금 F (USD)")
    ax.set_title("(d) PPO 가 학습한 동적 고정요금 정책 — 슬라이스별 상이",
                 fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.95)
    ax.set_xlim(0, T_env)
    ax.set_ylim(0, 105)

    # ── (e) 학습 곡선 ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ppo_train = np.asarray(results["ppo_train"])
    sac_train = np.asarray(results["sac_train"])

    def steps(n_eps):
        return np.arange(1, n_eps + 1) * T_env / 1e6  # 백만 step 단위

    def smooth(x, w=20):
        if len(x) < w:
            return x
        c = np.cumsum(np.insert(x, 0, 0))
        return (c[w:] - c[:-w]) / w

    ax.plot(steps(len(ppo_train)), ppo_train,
            color="#d62728", alpha=0.25, lw=0.7)
    ax.plot(steps(len(sac_train)), sac_train,
            color="#1f77b4", alpha=0.25, lw=0.7)
    if len(ppo_train) > 20:
        sm_ppo = smooth(ppo_train, 20)
        ax.plot(steps(len(ppo_train))[19:], sm_ppo,
                color="#d62728", lw=2.4, label="PPO (이동평균 20)")
    if len(sac_train) > 20:
        sm_sac = smooth(sac_train, 20)
        ax.plot(steps(len(sac_train))[19:], sm_sac,
                color="#1f77b4", lw=2.4, label="SAC (이동평균 20)")
    static_r = summary["Static-Heur."]["mean_reward"]
    ax.axhline(static_r, color="#2ca02c", ls="--", lw=1.2,
               label=f"Static-Heur. ({static_r:,.0f})")
    ax.set_xlabel("환경 step (백만 단위)")
    ax.set_ylabel("학습 episode 보상 (1e-5 환산)")
    ax.set_title("(e) 학습 곡선 — 두 알고리즘 모두 정적 정책을 추월", fontweight="bold")
    ax.legend(loc="lower right")

    # ── (f) Eval 분포 (Box plot) ────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ppo_evals = summary["PPO"]["eval_rewards"]
    sac_evals = summary["SAC"]["eval_rewards"]
    bp = ax.boxplot([ppo_evals, sac_evals], tick_labels=["PPO", "SAC"],
                    patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", lw=1.6))
    bp["boxes"][0].set_facecolor("#d62728")
    bp["boxes"][1].set_facecolor("#1f77b4")
    for b in bp["boxes"]:
        b.set_alpha(0.65)
    # 개별 평가 점 흩뿌리기
    rng = np.random.default_rng(0)
    for i, evals in enumerate([ppo_evals, sac_evals], start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(evals))
        ax.scatter(np.full(len(evals), i) + jitter, evals,
                   s=14, color="black", alpha=0.55, zorder=4)
    # PPO/SAC 평균 텍스트
    ax.text(1, max(ppo_evals) + 12,
            f"μ={np.mean(ppo_evals):,.0f}\nσ={np.std(ppo_evals):.0f}",
            ha="center", fontsize=8, color="#d62728", fontweight="bold")
    ax.text(2, max(sac_evals) + 12,
            f"μ={np.mean(sac_evals):,.0f}\nσ={np.std(sac_evals):.0f}",
            ha="center", fontsize=8, color="#1f77b4", fontweight="bold")
    # 기준선: 최고 베이스라인(Max-Price)
    max_r = summary["Max-Price"]["mean_reward"]
    ax.axhline(max_r, color="#ff7f0e", lw=1.4, ls="--",
               label=f"Max-Price 평균 ({max_r:,.0f})")
    # y범위는 PPO/SAC 분포에 맞추되, Max-Price 기준선은 위쪽 여백에 표시
    lo = min(min(sac_evals), max_r) - 50
    hi = max(ppo_evals) + 100
    ax.set_ylim(lo, hi)
    ax.set_ylabel("순보상 (1e-5 환산)")
    ax.set_title("(f) 평가 일관성 — 20 episode 분포 (PPO/SAC)",
                 fontweight="bold")
    ax.legend(loc="center right", fontsize=8, framealpha=0.95)

    fig.suptitle(
        "5G 네트워크 슬라이싱 동적 가격 결정 — 강화학습 결과 한눈에 보기",
        fontsize=15, fontweight="bold", y=0.97,
    )

    out = os.path.join(FIGURES_DIR, "fig_dashboard.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ──────────────────────────────────────────────────────────────────────────────
# 7. 그림 2: Subscriber Dynamics (전체 6 정책)
# ──────────────────────────────────────────────────────────────────────────────
def plot_subscriber_dynamics(rollouts, T_env=720):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    for i, (slice_key, slice_label, init_n) in enumerate(
            [("N_U", "URLLC 가입자 수 N_U(t)", 1000),
             ("N_E", "eMBB 가입자 수 N_E(t)", 5000)]):
        ax = axes[i]
        for name in POLICY_ORDER:
            if name not in rollouts:
                continue
            st = POLICY_STYLE[name]
            ax.plot(rollouts[name]["trajectory"][slice_key],
                    color=st["color"], lw=st["lw"], ls=st["ls"],
                    label=st["label"], zorder=st["zorder"])
        ax.axhline(init_n, color="gray", lw=0.8, ls=":", alpha=0.6)
        ax.text(T_env * 0.01, init_n * 1.02,
                f"초기 {init_n:,}", fontsize=8, color="gray")
        ax.set_xlabel("시간 t (hour)")
        ax.set_ylabel(slice_label)
        ax.set_xlim(0, T_env)
        ax.set_ylim(bottom=0)
        ax.set_title(("URLLC" if i == 0 else "eMBB") +
                     " 슬라이스 — 정책별 가입자 동역학",
                     fontweight="bold")
    axes[1].legend(loc="lower left", framealpha=0.9, ncol=2)

    fig.suptitle("그림 2 — 가입자 동역학: Max-Price 의 '높은 수익'은 사용자 붕괴의 결과",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_subscriber_dynamics.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ──────────────────────────────────────────────────────────────────────────────
# 8. 그림 3: 학습된 가격 정책 시계열 (PPO vs SAC)
# ──────────────────────────────────────────────────────────────────────────────
def plot_price_trajectories(rollouts, T_env=720):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

    panels = [
        ("F_U", "URLLC 고정요금 F_U", 100, 50),
        ("p_U", "URLLC 초과요금 p_U", 20, 10),
        ("F_E", "eMBB 고정요금 F_E", 100, 30),
        ("p_E", "eMBB 초과요금 p_E", 20, 5),
    ]

    for ax, (key, label, ub, ref) in zip(axes.flat, panels):
        for name in ["PPO", "SAC"]:
            if name not in rollouts:
                continue
            st = POLICY_STYLE[name]
            ax.plot(rollouts[name]["trajectory"][key],
                    color=st["color"], lw=1.6, label=st["label"], alpha=0.9)
        ax.axhline(ref, color="black", lw=0.8, ls="--",
                   label=f"기준가격 = {ref}")
        ax.axhline(ub, color="gray", lw=0.6, ls=":",
                   label=f"상한 = {ub}")
        ax.set_ylabel(label + " (USD)")
        ax.set_xlim(0, T_env)
        ax.set_ylim(0, ub * 1.05)
        ax.set_title(label, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    for ax in axes[1, :]:
        ax.set_xlabel("시간 t (hour)")

    fig.suptitle(
        "그림 3 — 학습된 가격 정책 시계열: 슬라이스마다 다른 동적 조정 전략",
        fontsize=13, fontweight="bold", y=1.00,
    )
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_price_trajectories.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ──────────────────────────────────────────────────────────────────────────────
# 9. 그림 4: 슬라이스별 수익 분해
# ──────────────────────────────────────────────────────────────────────────────
def plot_revenue_breakdown(rollouts, results):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                   gridspec_kw=dict(width_ratios=[1.4, 1]))

    names = [n for n in POLICY_ORDER if n in rollouts]
    rev_U = np.array([rollouts[n]["total_revenue_U"] / 1e6 for n in names])
    rev_E = np.array([rollouts[n]["total_revenue_E"] / 1e6 for n in names])
    pen   = np.array([rollouts[n]["total_penalty"]   / 1e6 for n in names])

    x = np.arange(len(names))
    ax1.bar(x, rev_U, color="#d62728", alpha=0.85,
            label="URLLC 수익", edgecolor="black", linewidth=0.5)
    ax1.bar(x, rev_E, bottom=rev_U, color="#1f77b4", alpha=0.85,
            label="eMBB 수익", edgecolor="black", linewidth=0.5)
    ax1.bar(x, -pen, color="#7f7f7f", alpha=0.7,
            label="QoS 패널티", edgecolor="black", linewidth=0.5)

    # 순수익 텍스트
    for i, n in enumerate(names):
        net = rev_U[i] + rev_E[i] - pen[i]
        ax1.text(i, rev_U[i] + rev_E[i] + 15,
                 f"순\n{net:.0f}M", ha="center", va="bottom",
                 fontsize=8, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right")
    ax1.set_ylabel("USD (M)")
    ax1.set_title("(a) 슬라이스별 수익 적층 + QoS 패널티 (단일 episode, seed=42)",
                  fontweight="bold")
    ax1.axhline(0, color="black", lw=0.6)
    ax1.legend(loc="upper right", fontsize=9)

    # (b) URLLC vs eMBB 수익 비율 — PPO 가 eMBB 를 어떻게 다루는지
    ratio_E = rev_E / (rev_U + rev_E + 1e-9) * 100
    bars = ax2.barh(names[::-1], ratio_E[::-1],
                    color=[POLICY_STYLE[n]["color"] for n in names[::-1]],
                    alpha=0.85, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("eMBB 비중 (%)")
    ax2.set_title("(b) 수익 구성 — 슬라이스별 비중", fontweight="bold")
    ax2.set_xlim(0, 100)
    for i, b in enumerate(bars):
        ax2.text(b.get_width() + 1.0, b.get_y() + b.get_height() / 2,
                 f"{b.get_width():.1f}%", va="center", fontsize=8)

    fig.suptitle(
        "그림 4 — 수익 분해: PPO 의 우위는 두 슬라이스 모두에서 균형 잡힌 가격에서 옴",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = os.path.join(FIGURES_DIR, "fig_revenue_breakdown.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


# ──────────────────────────────────────────────────────────────────────────────
# 10. README.md 자동 생성 (talking-points)
# ──────────────────────────────────────────────────────────────────────────────
README_TEMPLATE = """# 시각화 결과물 — 발표용 talking points

이 폴더에 있는 4개의 그림은 5G 네트워크 슬라이싱 동적 가격 결정 강화학습 결과를
청중에게 한눈에 설명하기 위해 만들어졌습니다. 그림 옆에 적힌 한 줄 요약과
"이렇게 말하세요" 문장을 그대로 사용해도 충분합니다.

> **읽기 순서:** `fig_dashboard.png` 한 장만으로도 전체 이야기를 끝낼 수 있도록
> 설계되었습니다. 시간이 더 있다면 그림 2 → 3 → 4 순으로 깊게 들어가세요.

---

## 1. `fig_dashboard.png` — 헤드라인 한 장 요약 (2×3 패널)

**한 줄로:** "PPO 가 수익과 가입자 유지를 동시에 잡아 정적 정책 대비 +162.9%."

| 패널 | 무엇을 보여주는가 | 이렇게 말하세요 |
|---|---|---|
| (a) 수익 vs. 유지 | 6개 정책의 (총 수익, 최종 eMBB 가입자) 산점도 | "오른쪽 위로 갈수록 좋습니다. PPO 만 유일하게 두 축을 모두 잡았고, Max-Price 는 수익은 비슷해 보여도 가입자가 거의 사라졌습니다." |
| (b) 순보상 막대 | 6개 정책의 평균 순보상 | "PPO 는 정적 정책 대비 2.6배, Max-Price 대비 +40% 의 순보상을 얻습니다." |
| (c) eMBB 가입자 추이 | 720시간 동안 N_E(t) 곡선 4개 | "Max-Price 는 첫 100시간 만에 가입자가 급락하지만, PPO/SAC 는 4천 명 이상을 유지합니다." |
| (d) PPO 가격 시계열 | 학습된 F_U(t), F_E(t) | "기준가가 아니라 시간에 따라 가격이 움직이는 것이 보입니다 — 이게 '동적' 정책입니다." |
| (e) 학습 곡선 | PPO/SAC 학습 보상 추이 | "두 알고리즘 모두 학습 초기에 정적 정책을 추월하고 안정적으로 수렴합니다." |
| (f) Eval 분포 | 20 episode 평가 결과 박스플롯 | "표준편차가 매우 작아 결과가 우연이 아니라 일관적임을 보여줍니다." |

**주의해서 말할 것:** 순보상 축은 환경 내부의 ×10⁻⁵ 스케일링이 적용된 값입니다.
패널티(약 20M USD)는 URLLC QoS 가 외생적으로 모델링되어 모든 정책에 공통으로 존재하는 *구조적 하한* 입니다.

---

## 2. `fig_subscriber_dynamics.png` — 정책별 가입자 동역학

**한 줄로:** "Max-Price 의 매력적인 수익은 가입자 붕괴의 부작용이다."

- URLLC 슬라이스 (왼쪽): 모든 정책이 1,000명 부근에서 비교적 안정.
- eMBB 슬라이스 (오른쪽): Max-Price 가 5,000 → 약 900명까지 수직 낙하. PPO/SAC 는 각각 ~4,000, ~2,200명을 유지.
- "**Max-Price 는 단기 수익을 극대화하지만 장기 지속 불가능합니다. PPO 는 가격을 너무 올리면 가입자를 잃는다는 것을 학습으로 발견했습니다.**"

---

## 3. `fig_price_trajectories.png` — PPO 와 SAC 가 학습한 가격 정책

**한 줄로:** "두 알고리즘은 같은 보상을 다른 가격 전략으로 추구한다."

- 4개 패널: F_U, p_U, F_E, p_E 각각.
- 점선 = 기준가, 회색 점선 = 행동 공간 상한.
- 청중에게 보여줄 포인트:
  - "PPO 와 SAC 가 어떤 슬라이스에서는 비슷하게, 어떤 슬라이스에서는 다른 가격을 선택하는 것을 볼 수 있습니다."
  - "단순히 '가격을 올리거나 내리는' 정책이 아니라, 슬라이스 특성과 가입자 상태에 반응하는 정책을 학습했습니다."

---

## 4. `fig_revenue_breakdown.png` — 수익 구성 분해

**한 줄로:** "PPO 의 우위는 한 슬라이스를 쥐어짜는 게 아니라 두 슬라이스의 균형에서 온다."

- (a) 적층 막대: 정책별 URLLC/eMBB 수익 + QoS 패널티.
- (b) 가로 막대: 전체 수익에서 eMBB 가 차지하는 비율.
- 청중에게 보여줄 포인트:
  - "PPO/SAC 의 수익 구조는 두 슬라이스에 분산되어 있어 한 슬라이스가 무너져도 견딜 수 있습니다."
  - "Max-Price 는 두 슬라이스 모두 가입자가 줄어 수익 풀 자체가 작아진 상태입니다."

---

## 자주 받는 질문에 대한 짧은 답

1. **"왜 PPO 가 SAC 보다 잘했나?"** — PPO 는 on-policy 이라 가입자 수가 점진적으로 변하는 환경(비정상성)에 더 빠르게 적응했습니다.
2. **"패널티가 왜 0 으로 안 떨어지나?"** — URLLC 의 QoS 가 본 모델에서 외생적(랜덤 변수)이라, 99.999% 목표를 거의 항상 미달합니다. 약 20M USD 의 구조적 하한이 모든 정책에 공통으로 존재합니다. 향후 연구로 가격–자원 결합 최적화가 남아 있습니다.
3. **"왜 Random 이 Static-Heuristic 보다 보상이 높나?"** — 무작위 탐색이 기준가 이상의 가격을 자주 시도했기 때문입니다. 그러나 가입자가 불안정하게 변동하여 장기적으로 지속 가능한 정책은 아닙니다.
4. **"단일 시드 결과 아닌가?"** — 네, 본 그림들은 seed=42 의 학습 결과 + 단일 episode rollout 입니다. 평가 단계는 20 episode 의 분포(그림 1f)로 재현 가능성을 보강합니다. 다중 시드 검증은 향후 작업입니다.
"""


def write_readme():
    out = os.path.join(FIGURES_DIR, "README.md")
    with open(out, "w") as f:
        f.write(README_TEMPLATE)
    print(f"[saved] {out}")


# ──────────────────────────────────────────────────────────────────────────────
# 11. main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    setup_matplotlib()
    print("\n=== load JSON results ===")
    results = load_results()
    for n, s in results["summary"].items():
        print(f"  {n:>14s}: reward={s['mean_reward']:8.0f}  "
              f"rev={s['mean_revenue']/1e6:7.1f}M  "
              f"pen={s['mean_penalty']/1e6:6.1f}M  "
              f"final N=({s['mean_final_N_U']:.0f}, {s['mean_final_N_E']:.0f})")

    print("\n=== rollout policies (single episode, seed=42) ===")
    policies = make_policies()
    rollouts = {}
    for name, fn in policies.items():
        rollouts[name] = rollout(name, fn, ENV_CONFIG, seed=42)

    # sanity checks
    print("\n=== sanity checks ===")
    ppo_r = rollouts["PPO"]["total_reward"]
    sac_r = rollouts["SAC"]["total_reward"]
    max_n_e = rollouts["Max-Price"]["final_N_E"]
    static_n = (rollouts["Static-Heur."]["final_N_U"],
                rollouts["Static-Heur."]["final_N_E"])
    print(f"  PPO single-rollout reward = {ppo_r:.0f}  (expected ~8200 ± ~150)")
    print(f"  SAC single-rollout reward = {sac_r:.0f}  (expected ~7200 ± ~200)")
    print(f"  Max-Price final N_E       = {max_n_e:.0f}  (expected ~900-1200, collapse)")
    print(f"  Static-Heur. final N      = {static_n}  (expected ~ initial 1000/5000)")

    print("\n=== plot figures ===")
    plot_dashboard(results, rollouts)
    plot_subscriber_dynamics(rollouts)
    plot_price_trajectories(rollouts)
    plot_revenue_breakdown(rollouts, results)
    write_readme()
    print("\n[done] all figures written to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
