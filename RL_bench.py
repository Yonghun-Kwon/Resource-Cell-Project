"""
Conv_BN_ReLU PPO Optimizer — ResNet-50 Ground Truth
=====================================================
State  : 파라미터(40d) + OI(8d) + Mem(8d) + Tput(8d) + reg(5d) → 69차원
Action : (B,C,H,W,K) 조정량 → 40차원 (8스텝 × 5파라미터)
Reward : 다변량 회귀(OI + Mem)가 ResNet-50 Conv 실측값 예측하는 MAPE 감소량

회귀 모델:
  log10(Tput) = β0 + β1·log10(OI) + β2·log10(Mem)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Tuple, List

import torchvision.models as tvm
from model_profiler import ModelProfiler

warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")

# ──────────────────────────────────────────
# 1. 상수 및 파라미터 공간
# ──────────────────────────────────────────

N_STEPS          = 8
N_PARAMS         = 5              # B, C, H, W, K
ACTION_DIM       = N_STEPS * N_PARAMS        # 40d
# params(40) + OI(8) + Mem(8) + Tput(8) + reg(5) = 69d
STATE_DIM        = N_STEPS * N_PARAMS + N_STEPS + N_STEPS + N_STEPS + 5
N_PROFILE_RUNS   = 5              # ResNet-50 측정 반복 횟수 (평균용)

B_MIN, B_MAX = 1,   4
C_MIN, C_MAX = 16,  512
H_MIN, H_MAX = 7,   224
W_MIN, W_MAX = 7,   224
K_CHOICES    = [1, 3]             # Conv kernel: 1×1 또는 3×3

PARAM_PENALTY_RATIO = 0.85
PARAM_PENALTY_COEF  = 0.05

# 초기 파라미터 (mdb.py Conv 설정 기반, 8개)
INIT_PARAMS = np.array([
    [1,  64,  112, 112, 3],
    [1,  64,   56,  56, 3],
    [1, 128,   56,  56, 3],
    [1, 128,   28,  28, 3],
    [1, 256,   28,  28, 3],
    [1, 256,   14,  14, 3],
    [1, 512,   14,  14, 3],
    [1, 512,    7,   7, 3],
], dtype=np.float32)

WARMUP = 1
RUNS   = 3


# ──────────────────────────────────────────
# 2. 물리 계산 (Conv_BN_ReLU)
# ──────────────────────────────────────────

def compute_flops_mem(B, C, H, W, K) -> Tuple[float, float]:
    """Conv(3×3)+BN+ReLU MACs / memory"""
    B, C, H, W, K = int(B), int(C), int(H), int(W), int(K)
    elems      = B * C * H * W
    elem_bytes = 4
    flops_conv = 2.0 * elems * C * K * K
    flops_bn   = 2.0 * elems
    flops_relu = elems
    flops      = flops_conv + flops_bn + flops_relu
    mem_conv   = (elems + C*C*K*K + elems) * elem_bytes
    mem_bn     = (elems + elems)            * elem_bytes
    mem_relu   = (elems + elems)            * elem_bytes
    mem        = mem_conv + mem_bn + mem_relu
    return float(flops), float(mem)


def compute_oi(flops, mem) -> float:
    return flops / max(mem, 1.0)


# ──────────────────────────────────────────
# 3. Conv 실측
# ──────────────────────────────────────────

def measure_conv_throughput(B, C, H, W, K) -> float:
    B, C, H, W, K = int(B), int(C), int(H), int(W), int(K)
    conv = nn.Conv2d(C, C, kernel_size=K, padding=K//2, bias=False).eval()
    bn   = nn.BatchNorm2d(C).eval()
    x    = torch.randn(B, C, H, W)
    fn   = lambda: F.relu(bn(conv(x)))
    for _ in range(WARMUP):
        fn()
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    lat_s = float(np.mean(times))
    flops, _ = compute_flops_mem(B, C, H, W, K)
    return (flops / 1e9) / max(lat_s, 1e-9)


# ──────────────────────────────────────────
# 4. ResNet-50 Ground Truth 수집
# ──────────────────────────────────────────

def collect_resnet50_ground_truth(n_runs: int = N_PROFILE_RUNS):
    """
    ResNet-50 Conv2d 레이어를 n_runs 회 프로파일링 후
    latency 평균으로 고정 ground truth 생성.
    Returns: ndarray (N, 3) — col0=OI, col1=Mem(bytes), col2=Tput(GFLOPS)
    """
    print(f"\n[GT] ResNet-50 Conv2d 프로파일링 ({n_runs}회 평균)...")
    model = tvm.resnet50(weights=None).eval()
    x     = torch.randn(1, 3, 224, 224)

    all_lats = {}
    all_meta = {}   # layer_name → (oi, macs, mem_bytes)

    for run in range(n_runs):
        p = ModelProfiler(target_ops=[nn.Conv2d], warmup=2, runs=5)
        p.profile(model, x)
        for prof in p.profiles:
            if prof.layer_name not in all_lats:
                all_lats[prof.layer_name] = []
                all_meta[prof.layer_name] = (prof.oi, prof.macs, prof.mem_bytes)
            all_lats[prof.layer_name].append(prof.latency_ms)
        print(f"  run {run+1}/{n_runs} 완료")

    gt = []
    for name, lats in all_lats.items():
        oi, macs, mem = all_meta[name]
        avg_lat_ms = float(np.mean(lats))
        tput = (macs / 1e9) / (avg_lat_ms / 1000) if avg_lat_ms > 1e-9 else 0.0
        if oi > 0 and tput > 0 and mem > 0:
            gt.append((oi, mem, tput))   # 3컬럼

    gt_arr = np.array(gt)   # (N, 3)
    print(f"[GT] 수집 완료: {len(gt_arr)}개 레이어  "
          f"OI [{gt_arr[:,0].min():.2f}, {gt_arr[:,0].max():.2f}]  "
          f"Mem [{gt_arr[:,1].min()/1e6:.1f}, {gt_arr[:,1].max()/1e6:.1f}] MB")
    return gt_arr


# ──────────────────────────────────────────
# 5. 회귀 모델
# ──────────────────────────────────────────

def fit_regression(ois, mems, tputs) -> Tuple[float, float, float, float, float]:
    """
    log10(Tput) = β0 + β1·log10(OI) + β2·log10(Mem)
    Returns: (β1, β2, β0, R², MAPE%)
    """
    mask = (ois > 0) & (mems > 0) & (tputs > 0)
    if mask.sum() < 3:
        return 0.0, 0.0, 0.0, -1.0, 100.0

    log_oi   = np.log10(ois[mask])
    log_mem  = np.log10(mems[mask])
    log_tput = np.log10(tputs[mask])

    # 설계 행렬 [1, log_OI, log_Mem]
    X = np.column_stack([np.ones(mask.sum()), log_oi, log_mem])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, log_tput, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0, -1.0, 100.0

    b0, b1, b2 = coeffs
    log_pred  = X @ coeffs
    ss_res    = np.sum((log_tput - log_pred) ** 2)
    ss_tot    = np.sum((log_tput - log_tput.mean()) ** 2)
    r2        = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
    pred_tput = 10 ** log_pred
    mape      = float(np.mean(np.abs(pred_tput - tputs[mask]) /
                               np.maximum(tputs[mask], 1e-9)) * 100)
    return float(b1), float(b2), float(b0), float(r2), mape


def predict_mape_on_gt(b1, b2, b0, gt: np.ndarray) -> float:
    """
    다변량 회귀로 ResNet-50 GT 예측 → MAPE
    gt: (N, 3) — col0=OI, col1=Mem, col2=Tput
    """
    if b1 == 0.0 and b2 == 0.0 and b0 == 0.0:
        return 100.0
    gt_ois  = gt[:, 0]
    gt_mems = gt[:, 1]
    gt_tput = gt[:, 2]
    log_pred = b0 + b1 * np.log10(np.maximum(gt_ois,  1e-9)) \
                  + b2 * np.log10(np.maximum(gt_mems, 1e-9))
    pred = 10 ** log_pred
    mape = float(np.mean(np.abs(pred - gt_tput) /
                          np.maximum(gt_tput, 1e-9)) * 100)
    return min(mape, 500.0)


# ──────────────────────────────────────────
# 6. 환경
# ──────────────────────────────────────────

class ConvEnv:
    """
    Conv_BN_ReLU PPO 환경

    State (44d):
      [0:40]  파라미터 (B,C,H,W,K)×8 정규화
      [40:48] OI log10 정규화
      [48:56] Tput log10 정규화  → 실제 44d: N_STEPS*5+N_STEPS+N_STEPS+4
      [40:44] reg (R², slope, intercept, GT_MAPE)

    Reward: GT_MAPE 감소량 (ResNet-50 예측 오차 감소)
    """

    def __init__(self, gt: np.ndarray, max_steps: int = 15,
                 action_scale: float = 16.0):
        self.gt           = gt        # (53, 2) — 고정 ground truth
        self.max_steps    = max_steps
        self.action_scale = action_scale
        self.reset()

    def reset(self) -> np.ndarray:
        self.params    = INIT_PARAMS.copy()
        self.step_cnt  = 0
        self.prev_mape = 100.0
        self._remeasure_all(prev_params=None)
        b1, b2, b0, _, _ = fit_regression(self.ois, self.mems, self.tputs)
        self.prev_mape = predict_mape_on_gt(b1, b2, b0, self.gt)
        return self._get_state()

    def _remeasure_all(self, prev_params=None):
        """변경된 스텝만 재측정, 나머지는 캐시 유지"""
        if not hasattr(self, 'ois') or prev_params is None:
            self.ois   = np.zeros(N_STEPS)
            self.mems  = np.zeros(N_STEPS)
            self.tputs = np.zeros(N_STEPS)
            prev_params = np.full_like(self.params, -1)  # 전체 강제 측정

        for i, (B, C, H, W, K) in enumerate(self.params):
            K = 3 if K >= 2 else 1
            # 이전과 동일한 파라미터면 캐시 사용
            if np.array_equal(self.params[i], prev_params[i]):
                continue
            flops, mem    = compute_flops_mem(B, C, H, W, K)
            self.ois[i]   = compute_oi(flops, mem)
            self.mems[i]  = mem
            self.tputs[i] = measure_conv_throughput(B, C, H, W, K)

    def _get_state(self) -> np.ndarray:
        # 파라미터 정규화 (40d)
        norm = self.params.copy().astype(np.float32)
        norm[:, 0] = (norm[:, 0] - B_MIN) / max(B_MAX - B_MIN, 1)
        norm[:, 1] = (norm[:, 1] - C_MIN) / (C_MAX - C_MIN)
        norm[:, 2] = (norm[:, 2] - H_MIN) / (H_MAX - H_MIN)
        norm[:, 3] = (norm[:, 3] - W_MIN) / (W_MAX - W_MIN)
        norm[:, 4] = (norm[:, 4] - 1)     / 2.0
        param_feat = norm.flatten()

        # OI log10 정규화 (8d) — 범위 1~1000
        oi_feat  = np.clip(np.log10(np.maximum(self.ois,  1e-9)) / 3.0, 0, 1)
        # Mem log10 정규화 (8d) — 범위 1KB~1GB → log10: 3~9
        mem_feat = np.clip((np.log10(np.maximum(self.mems, 1e-9)) - 3.0) / 6.0, 0, 1)
        # Tput log10 정규화 (8d)
        tput_feat = np.clip(np.log10(np.maximum(self.tputs, 1e-9)) / 3.0, 0, 1)

        # 회귀 품질 (5d): R², β1(OI), β2(Mem), β0, GT_MAPE
        b1, b2, b0, r2, _ = fit_regression(self.ois, self.mems, self.tputs)
        gt_mape = predict_mape_on_gt(b1, b2, b0, self.gt)
        reg_feat = np.array([
            np.clip(r2,              -1.0, 1.0),
            np.clip(b1 / 2.0,       -1.0, 1.0),
            np.clip(b2 / 2.0,       -1.0, 1.0),
            np.clip(b0 / 4.0,       -1.0, 1.0),
            np.clip(gt_mape / 100.0, 0.0, 1.0),
        ], dtype=np.float32)

        return np.concatenate([param_feat, oi_feat, mem_feat, tput_feat, reg_feat]).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        delta = action.reshape(N_STEPS, N_PARAMS) * self.action_scale

        new_params = self.params + delta
        new_params[:, 0] = np.clip(np.round(new_params[:, 0]), B_MIN, B_MAX)
        new_params[:, 1] = np.clip(np.round(new_params[:, 1]), C_MIN, C_MAX)
        new_params[:, 2] = np.clip(np.round(new_params[:, 2]), H_MIN, H_MAX)
        new_params[:, 3] = np.clip(np.round(new_params[:, 3]), W_MIN, W_MAX)
        new_params[:, 4] = np.where(new_params[:, 4] >= 2, 3, 1).astype(float)

        # OI 기준 오름차순 정렬
        ois_tmp = np.array([
            compute_oi(*compute_flops_mem(*p)) for p in new_params
        ])
        new_params = new_params[np.argsort(ois_tmp)]

        changed = not np.array_equal(new_params, self.params)
        prev_params = self.params.copy()
        self.params = new_params
        self.step_cnt += 1

        if changed:
            self._remeasure_all(prev_params=prev_params)

        b1, b2, b0, r2, _ = fit_regression(self.ois, self.mems, self.tputs)
        gt_mape = predict_mape_on_gt(b1, b2, b0, self.gt)

        # 보상: GT MAPE 감소량
        reward = (self.prev_mape - gt_mape) / 100.0
        if not changed:
            reward -= 0.1

        # soft penalty: C 또는 H,W가 상한 근처
        threshold = C_MAX * PARAM_PENALTY_RATIO
        excess    = np.maximum(self.params[:, 1] - threshold, 0.0)
        reward   -= PARAM_PENALTY_COEF * float(excess.mean() / C_MAX)

        self.prev_mape = gt_mape
        done = self.step_cnt >= self.max_steps
        return self._get_state(), float(reward), done

    def get_mape(self) -> float:
        return self.prev_mape

    def get_params(self) -> np.ndarray:
        return self.params.copy()

    def get_regression_info(self):
        return fit_regression(self.ois, self.mems, self.tputs)


# ──────────────────────────────────────────
# 7. PPO Actor-Critic
# ──────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),   nn.Tanh(),
        )
        self.actor_mean   = nn.Linear(hidden, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        self.critic       = nn.Linear(hidden, 1)

    def forward(self, x):
        feat  = self.encoder(x)
        mean  = torch.tanh(self.actor_mean(feat))
        std   = self.actor_logstd.exp().expand_as(mean)
        value = self.critic(feat).squeeze(-1)
        return mean, std, value

    def get_action(self, state, device):
        s = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            mean, std, value = self(s)
        dist   = torch.distributions.Normal(mean, std)
        action = torch.clamp(dist.sample(), -1.0, 1.0)
        logp   = dist.log_prob(action).sum(-1)
        return action.squeeze(0).cpu().numpy(), logp.item(), value.item()

    def evaluate(self, states, actions):
        mean, std, values = self(states)
        dist    = torch.distributions.Normal(mean, std)
        logps   = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logps, values, entropy


# ──────────────────────────────────────────
# 8. PPO 에이전트
# ──────────────────────────────────────────

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_eps=0.2, epochs=4, batch_size=32):
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps   = clip_eps
        self.epochs     = epochs
        self.batch_size = batch_size
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net        = ActorCritic(state_dim, action_dim).to(self.device)
        self.opt        = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.states:  List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float]      = []
        self.values:  List[float]      = []
        self.logps:   List[float]      = []
        self.dones:   List[bool]       = []

    def get_action(self, state):
        return self.net.get_action(state, self.device)

    def store(self, s, a, r, v, lp, done):
        self.states.append(s);  self.actions.append(a)
        self.rewards.append(r); self.values.append(v)
        self.logps.append(lp);  self.dones.append(done)

    def _compute_gae(self, last_value):
        rewards = np.array(self.rewards)
        values  = np.array(self.values + [last_value])
        dones   = np.array(self.dones, dtype=np.float32)
        advs    = np.zeros_like(rewards)
        gae     = 0.0
        for t in reversed(range(len(rewards))):
            delta  = rewards[t] + self.gamma*values[t+1]*(1-dones[t]) - values[t]
            gae    = delta + self.gamma*self.gae_lambda*(1-dones[t])*gae
            advs[t] = gae
        return advs, advs + values[:-1]

    def update(self, last_value):
        advs, rets = self._compute_gae(last_value)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        states  = torch.FloatTensor(np.stack(self.states)).to(self.device)
        actions = torch.FloatTensor(np.stack(self.actions)).to(self.device)
        old_lps = torch.FloatTensor(np.array(self.logps)).to(self.device)
        advs_t  = torch.FloatTensor(advs).to(self.device)
        rets_t  = torch.FloatTensor(rets).to(self.device)
        total_loss = 0.0
        n = len(self.states)
        for _ in range(self.epochs):
            for start in range(0, n, self.batch_size):
                mb = np.random.choice(n, min(self.batch_size, n), replace=False)
                new_lps, vals, entropy = self.net.evaluate(states[mb], actions[mb])
                ratio = (new_lps - old_lps[mb]).exp()
                clip  = ratio.clamp(1-self.clip_eps, 1+self.clip_eps)
                loss  = (-torch.min(ratio*advs_t[mb], clip*advs_t[mb]).mean()
                         + 0.5*F.mse_loss(vals, rets_t[mb])
                         - 0.01*entropy.mean())
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                total_loss += loss.item()
        self.states.clear(); self.actions.clear(); self.rewards.clear()
        self.values.clear(); self.logps.clear(); self.dones.clear()
        return total_loss


# ──────────────────────────────────────────
# 9. 학습 루프
# ──────────────────────────────────────────

def train(gt: np.ndarray, n_episodes=200, max_steps=15,
          action_scale=16.0, rollout_len=45, verbose=True):

    env   = ConvEnv(gt, max_steps=max_steps, action_scale=action_scale)
    agent = PPOAgent(STATE_DIM, ACTION_DIM)

    print(f"\n{'='*58}")
    print(f"  Conv PPO Optimizer  (ResNet-50 GT: {len(gt)} layers)")
    print(f"  State {STATE_DIM}d | Action {ACTION_DIM}d | Device: {agent.device}")
    print(f"  Episodes: {n_episodes}  Steps/ep: {max_steps}")
    print(f"{'='*58}\n")

    ep_mapes  = []
    ep_rewards = []
    losses     = []
    rollout_step = 0

    for ep in range(1, n_episodes + 1):
        state     = env.reset()
        ep_reward = 0.0

        for _ in range(max_steps):
            action, logp, value = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.store(state, action, reward, value, logp, done)
            ep_reward    += reward
            rollout_step += 1
            state = next_state

            if rollout_step >= rollout_len or done:
                _, _, lv = agent.get_action(state)
                losses.append(agent.update(lv if not done else 0.0))
                rollout_step = 0
            if done:
                break

        ep_mapes.append(env.get_mape())
        ep_rewards.append(ep_reward)

        if verbose and ep % 20 == 0:
            print(f"  Ep {ep:>4}/{n_episodes} | "
                  f"Reward={np.mean(ep_rewards[-20:]):+.4f} | "
                  f"GT_MAPE(last)={ep_mapes[-1]:7.2f}% | "
                  f"GT_MAPE(avg20)={np.mean(ep_mapes[-20:]):7.2f}%")

    print(f"\n{'='*58}")
    print(f"  학습 완료  |  최종 GT MAPE = {ep_mapes[-1]:.2f}%")
    b1, b2, b0, r2, bench_mape = env.get_regression_info()
    print(f"  회귀: log10(T) = {b0:.4f} + {b1:.4f}·log10(OI) + {b2:.4f}·log10(Mem)")
    print(f"        R²={r2:.4f}  Bench MAPE={bench_mape:.2f}%")
    print(f"  최적 파라미터:")
    params = env.get_params()
    print(f"  {'#':>2} {'B':>3} {'C':>5} {'H':>5} {'W':>5} {'K':>3}"
          f"  {'OI':>8}  {'Tput(GFLOPS)':>13}")
    print(f"  {'-'*52}")
    for i, (B, C, H, W, K) in enumerate(params, 1):
        B,C,H,W,K = int(B),int(C),int(H),int(W),int(K)
        flops, mem = compute_flops_mem(B,C,H,W,K)
        oi   = compute_oi(flops, mem)
        tput = measure_conv_throughput(B,C,H,W,K)
        print(f"  {i:>2} {B:>3} {C:>5} {H:>5} {W:>5} {K:>3}"
              f"  {oi:>8.3f}  {tput:>13.4f}")
    print(f"{'='*58}\n")

    return agent, env, {"mapes": ep_mapes, "rewards": ep_rewards, "losses": losses}


# ──────────────────────────────────────────
# 10. 시각화
# ──────────────────────────────────────────

def plot_results(history, gt, env, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#0f0f1a")

    def style(ax, title, ylabel, color):
        ax.set_facecolor("#1a1a2e")
        ax.set_title(title, color="white", fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, color=color, fontsize=9)
        ax.set_xlabel("Episode", color="#aaaaaa", fontsize=9)
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.grid(True, color="#2a2a4a", linewidth=0.5)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333355")

    w = 20
    mapes   = history["mapes"]
    rewards = history["rewards"]
    losses  = history["losses"]

    # GT MAPE 추이
    sm = np.convolve(mapes, np.ones(w)/w, mode="valid")
    axes[0].plot(mapes,  color="#f0c040", alpha=0.2, linewidth=0.8)
    axes[0].plot(range(w-1, len(mapes)), sm, color="#f0c040", linewidth=1.8)
    axes[0].axhline(0, color="#ff6b6b", linewidth=1, linestyle="--", alpha=0.5)
    style(axes[0], "ResNet-50 GT MAPE (%)", "MAPE (%)", "#f0c040")

    # 보상 추이
    sm_r = np.convolve(rewards, np.ones(w)/w, mode="valid")
    axes[1].plot(rewards, color="#4ecdc4", alpha=0.2, linewidth=0.8)
    axes[1].plot(range(w-1, len(rewards)), sm_r, color="#4ecdc4", linewidth=1.8)
    style(axes[1], "Episode Reward", "Reward", "#4ecdc4")

    # 최종 회귀선 vs GT scatter
    b1, b2, b0, r2, _ = env.get_regression_info()
    oi_range  = np.logspace(np.log10(max(gt[:,0].min(), 1e-1)),
                            np.log10(gt[:,0].max()), 100)
    mem_mean  = np.mean(np.log10(np.maximum(gt[:,1], 1e-9)))  # Mem 평균값으로 고정
    pred      = 10 ** (b0 + b1*np.log10(oi_range) + b2*mem_mean)
    axes[2].scatter(gt[:,0], gt[:,2], color="#aaaaaa", s=20, alpha=0.6, label="ResNet-50 GT")
    axes[2].scatter(env.ois, env.tputs, color="#f0c040", s=40, zorder=5, label="Bench params")
    axes[2].plot(oi_range, pred, color="#ff6b6b", linewidth=1.5, label=f"Reg R²={r2:.3f}")
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_xlabel("OI (MACs/byte)", color="#aaaaaa", fontsize=9)
    axes[2].set_ylabel("Throughput (GFLOPS)", color="#aaaaaa", fontsize=9)
    axes[2].legend(fontsize=8, facecolor="#0f0f1a", edgecolor="gray", labelcolor="white")
    style(axes[2], "Regression vs GT", "Throughput (GFLOPS)", "#ff6b6b")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Plot saved → {save_path}")


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("ppo_results", exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    # Step 1: ResNet-50 ground truth 수집 (고정)
    gt = collect_resnet50_ground_truth(n_runs=N_PROFILE_RUNS)
    np.save("ppo_results/resnet50_gt.npy", gt)
    print(f"GT saved → ppo_results/resnet50_gt.npy")

    # Step 2: PPO 학습
    agent, env, history = train(
        gt           = gt,
        n_episodes   = 200,
        max_steps    = 15,
        action_scale = 16.0,
        rollout_len  = 45,
        verbose      = True,
    )

    # Step 3: 결과 저장
    plot_results(history, gt, env, "ppo_results/conv_training.png")
    torch.save(agent.net.state_dict(), "ppo_results/conv_actor_critic.pt")
    print("Model saved → ppo_results/conv_actor_critic.pt")