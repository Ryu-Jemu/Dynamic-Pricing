#!/usr/bin/env python3
"""
5G O-RAN Network Slicing - Optimized CMDP-SAC Training
======================================================

맥북 MPS 가속, tqdm 진행률 표시, 계산량 최적화 적용

한 번에 실행:
    pip install -r requirements.txt && python run_training.py -t 100000 -f

Author: Research Team
Date: 2026-01
"""

import os
import sys
import time
import json
import argparse
import warnings
from datetime import datetime
from typing import Dict, Tuple

import numpy as np

# 경고 숨기기
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 프로젝트 경로
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def get_device() -> str:
    """맥북 MPS 가속 자동 감지"""
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS 테스트
            try:
                x = torch.zeros(1, device='mps')
                del x
                return "mps"
            except:
                pass
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except:
        return "cpu"


def create_env(fast_mode: bool = False):
    """최적화된 환경 생성"""
    from config.scenario_config import ScenarioConfig
    from env.network_slicing_cmdp_env import NetworkSlicingCMDPEnv
    return NetworkSlicingCMDPEnv(config=ScenarioConfig())


def train(
    total_timesteps: int = 100000,
    device: str = "auto",
    fast_mode: bool = False,
    log_dir: str = "./logs",
    seed: int = 42
) -> Tuple[object, Dict]:
    """CMDP-SAC 학습 실행"""
    
    # 지연 임포트 (시작 시간 단축)
    import torch
    from tqdm import tqdm
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    
    # 헤더 출력
    print("\n" + "=" * 65)
    print("  🚀 5G O-RAN Network Slicing - CMDP-SAC Training")
    print("=" * 65)
    
    # 디바이스 설정
    if device == "auto":
        device = get_device()
    
    device_emoji = {"mps": "🍎", "cuda": "🎮", "cpu": "💻"}.get(device, "💻")
    print(f"  {device_emoji} Device: {device.upper()}")
    
    # 로그 디렉토리
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"  📁 Log dir: {run_dir}")
    
    # 환경 생성
    print("  🌐 Creating environment...", end=" ", flush=True)
    env = create_env(fast_mode)
    eval_env = create_env(fast_mode)
    print(f"Done (obs:{env.observation_space.shape[0]}, act:{env.action_space.shape[0]})")
    
    # SAC 모델
    print("  🧠 Building SAC model...", end=" ", flush=True)
    
    policy_kwargs = {
        "net_arch": [256, 256] if fast_mode else [256, 256, 128],
        "activation_fn": torch.nn.ReLU
    }
    
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=100000 if fast_mode else 200000,
        batch_size=256 if fast_mode else 512,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        learning_starts=1000 if fast_mode else 2000,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=policy_kwargs,
        device=device,
        seed=seed,
        verbose=0
    )
    print("Done")
    
    # tqdm 콜백
    class ProgressCallback(BaseCallback):
        def __init__(self, total: int):
            super().__init__()
            self.total = total
            self.pbar = None
            self.metrics = {'profit': [], 'u_viol': [], 'e_viol': []}
            
        def _on_training_start(self):
            self.pbar = tqdm(
                total=self.total, desc="  📈 Training", unit="step",
                ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
            self.t0 = time.time()
            
        def _on_step(self) -> bool:
            self.pbar.update(1)
            if self.n_calls % 100 == 0:
                info = self.locals.get('infos', [{}])[0]
                for k, v in [('profit', 'profit'), ('u_viol', 'urllc_violation_rate'), ('e_viol', 'embb_violation_rate')]:
                    if v in info:
                        self.metrics[k].append(info[v])
                if self.metrics['profit']:
                    self.pbar.set_postfix({
                        'profit': f"${np.mean(self.metrics['profit'][-50:]):.0f}",
                        'U': f"{np.mean(self.metrics['u_viol'][-50:]):.3f}",
                        'E': f"{np.mean(self.metrics['e_viol'][-50:]):.3f}"
                    })
            return True
        
        def _on_training_end(self):
            self.pbar.close()
            print(f"  ⏱️  Elapsed: {(time.time()-self.t0)/60:.1f} min")
    
    # 콜백 설정
    progress_cb = ProgressCallback(total_timesteps)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(10000, total_timesteps // 5),
        save_path=run_dir, name_prefix="ckpt"
    )
    
    # 설정 저장
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump({"timesteps": total_timesteps, "device": device, "fast": fast_mode, "seed": seed}, f)
    
    print(f"  📊 Timesteps: {total_timesteps:,} (~{total_timesteps//168} episodes)")
    print("=" * 65 + "\n")
    
    # 학습
    start = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=[progress_cb, checkpoint_cb], progress_bar=False)
    except KeyboardInterrupt:
        print("\n  ⚠️  Interrupted")
    
    # 저장
    model.save(os.path.join(run_dir, "final_model"))
    print(f"\n  💾 Model saved: {run_dir}/final_model")
    
    # 평가
    print("\n" + "=" * 65)
    print("  📊 Evaluation (5 episodes)")
    print("=" * 65)
    
    results = evaluate(model, eval_env)
    
    with open(os.path.join(run_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 요약
    elapsed = time.time() - start
    print(f"""
  ┌{'─'*61}┐
  │  📈 Mean Reward: {results['mean_reward']:>8.2f} ± {results['std_reward']:.2f}{' '*20}│
  │  💰 Mean Profit: ${results['mean_profit']:>7.0f}{' '*32}│
  │  🔴 URLLC Viol:  {results['mean_urllc_violation']:>8.4f} (target: 0.001){' '*14}│
  │  🔵 eMBB Viol:   {results['mean_embb_violation']:>8.4f} (target: 0.01){' '*15}│
  │  ✅ Constraint:  {results['constraint_satisfaction']*100:>7.1f}%{' '*31}│
  │  ⏱️  Total Time: {elapsed/60:>7.1f} min ({total_timesteps/elapsed:.0f} steps/sec){' '*10}│
  └{'─'*61}┘
""")
    
    return model, results


def evaluate(model, env, n_episodes: int = 5) -> Dict:
    """모델 평가"""
    from tqdm import trange
    
    rewards, profits, u_viols, e_viols = [], [], [], []
    
    for _ in trange(n_episodes, desc="  🔍 Eval", ncols=60):
        obs, _ = env.reset()
        done, ep_r, ep_p, ep_u, ep_e = False, 0, 0, [], []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            ep_r += r
            ep_p += info.get('profit', 0)
            ep_u.append(info.get('urllc_violation_rate', 0))
            ep_e.append(info.get('embb_violation_rate', 0))
        
        rewards.append(ep_r)
        profits.append(ep_p)
        u_viols.append(np.mean(ep_u))
        e_viols.append(np.mean(ep_e))
    
    sat = sum(1 for u, e in zip(u_viols, e_viols) if u <= 0.001 and e <= 0.01) / n_episodes
    
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_profit": float(np.mean(profits)),
        "mean_urllc_violation": float(np.mean(u_viols)),
        "mean_embb_violation": float(np.mean(e_viols)),
        "constraint_satisfaction": float(sat)
    }


def main():
    parser = argparse.ArgumentParser(
        description="5G O-RAN CMDP-SAC Training (MPS Accelerated)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  한 번에 실행 (One-liner):
  
    pip install -r requirements.txt && python run_training.py -t 100000 -f
    
  예시:
    python run_training.py                    # 기본 (100K steps)
    python run_training.py -t 10000 -f        # 빠른 테스트 (10K)
    python run_training.py -t 500000          # 전체 학습 (500K)
    python run_training.py -d cpu             # CPU 강제
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
    )
    
    parser.add_argument("-t", "--timesteps", type=int, default=100000, help="학습 스텝 (default: 100000)")
    parser.add_argument("-d", "--device", type=str, default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("-f", "--fast", action="store_true", help="빠른 모드 (네트워크 축소)")
    parser.add_argument("-l", "--log-dir", type=str, default="./logs")
    parser.add_argument("-s", "--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    train(
        total_timesteps=args.timesteps,
        device=args.device,
        fast_mode=args.fast,
        log_dir=args.log_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
