#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_dimensio_benchmark_torch_mobile.py
PyTorch Mobile(TorchScript + optimize_for_mobile) 변환 버전

주요 변경점:
  - torch.jit.trace 로 모델 → TorchScript 변환
  - optimize_for_mobile 적용 (USE_MOBILE_OPTIMIZER=True)
  - inplace=True → inplace=False (모바일 호환성)
  - torch.cuda.synchronize() 제거 (모바일 CPU 타겟)
  - REBUILD_MODEL_EVERY_RUN 기본값 False (변환 오버헤드 방지)
  - SAVE_PTL=True 시 .ptl 파일로 배포용 저장 가능
"""
import torch, time
import torch.nn as nn
import math
import numpy as np

# PyTorch Mobile 최적화 임포트
try:
    from torch.utils.mobile_optimizer import optimize_for_mobile
    HAS_MOBILE_OPTIMIZER = True
except ImportError:
    print("[경고] torch.utils.mobile_optimizer 없음 — USE_MOBILE_OPTIMIZER 비활성화")
    HAS_MOBILE_OPTIMIZER = False

# ============================================================
# 전역 설정
# ============================================================
DEVICE = "cpu"
RUNS = 128
N_STEPS = 10
BYTES_PER = 4

# 모바일: 모델 재빌드 + 재변환 오버헤드가 크므로 False 권장
REBUILD_MODEL_EVERY_RUN = False

USE_MOBILE_OPTIMIZER = True   # optimize_for_mobile 적용 여부
SAVE_PTL = False              # True이면 .ptl 파일 저장 (실제 모바일 배포용)
PTL_SAVE_DIR = "."            # .ptl 저장 디렉터리

# ============================================================
# 계단형 비율(층 수 제어)
# ============================================================
RATIOS = {
    "conv":       [1],
    "dw":         [1],
    "gemm":       [1],
    "attn":       [1],
    "ln_attn_ln": [1],
}

# ============================================================
# 크기 설정 (Fit ~ Miss)
# ============================================================
SIZES_FIT = {
    "gemm":       dict(M=256,  K=256,  N=256),
    "conv":       dict(B=1, Cin=6,  Cout=12, H=224, W=224, K=3),
    "dw":         dict(B=1, C=8,   H=64,  W=64,  K=3),
    "vit":        dict(B=1, T=57,  H=4,   Dh=64),
    "ln_attn_ln": dict(B=1, T=160, H=4,   Dh=64),
}

SIZES_MISS = {
    "gemm":       dict(M=1024, K=1024, N=1024),
    "conv":       dict(B=1, Cin=24, Cout=48, H=224, W=224, K=3),
    "dw":         dict(B=1, C=96,  H=256, W=256, K=3),
    "vit":        dict(B=1, T=632, H=8,   Dh=64),
    "ln_attn_ln": dict(B=1, T=632, H=8,   Dh=64),
}

# ============================================================
# FLOPs / Bytes 유틸
# ============================================================
def flops_conv3x3(B, Cin, Cout, H, W, K=3): return 2*B*Cout*H*W*Cin*K*K
def bytes_conv3x3(B, Cin, Cout, H, W, K=3): return (B*Cin*H*W + B*Cout*H*W + Cout*Cin*K*K)*BYTES_PER
def flops_dw(B, C, H, W, K=3):  return 2*B*C*H*W*K*K
def bytes_dw(B, C, H, W, K=3):  return (B*C*H*W + B*C*H*W + C*K*K)*BYTES_PER
def flops_gemm(M, K, N):         return 2*M*K*N
def bytes_gemm(M, K, N):         return (M*K + K*N + 2*M*N)*BYTES_PER

def flops_attn_total(B, T, H, Dh):
    D = H * Dh
    proj = 8.0 * B * T * D * D
    core = 4.0 * B * H * T * T * Dh
    return proj + core

def flops_bytes_layernorm(B, T, D):
    f = 5.0 * B * T * D
    b = ((B*T*D) + (B*T*D) + 2*D) * BYTES_PER
    return f, b

def bytes_attn_total(B, T, H, Dh):
    D = H * Dh
    x_io   = (B*T*D + B*T*D)
    qkv_o  = (3*B*T*D + B*T*D)
    scores = (B*H*T*T) * 2
    return (x_io + qkv_o + scores) * BYTES_PER

# ============================================================
# PyTorch Mobile 변환 헬퍼
# ============================================================
def to_mobile_model(net: nn.Module,
                    example_input: torch.Tensor,
                    save_name: str = None) -> torch.jit.ScriptModule:
    """
    1) torch.jit.trace → TorchScript
    2) optimize_for_mobile (USE_MOBILE_OPTIMIZER=True 시)
    3) SAVE_PTL=True 시 .ptl 로 저장 (모바일 배포용)
    """
    net.eval()
    with torch.no_grad():
        scripted = torch.jit.trace(net, example_input)

    if USE_MOBILE_OPTIMIZER and HAS_MOBILE_OPTIMIZER:
        scripted = optimize_for_mobile(scripted)

    if SAVE_PTL and save_name:
        import os
        ptl_path = os.path.join(PTL_SAVE_DIR, f"{save_name}.ptl")
        scripted._save_for_lite_interpreter(ptl_path)
        print(f"  [저장] {ptl_path}")

    return scripted

def run_model(model, x, runs=RUNS):
    """TorchScript/일반 모델 모두 동일하게 호출 가능"""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model(x)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times)

# ============================================================
# 모듈 정의
# ============================================================

class PureSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.h  = n_head
        self.dh = d_model // n_head
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Dh = self.h, self.dh
        q = self.q(x).view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()
        k = self.k(x).view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()
        v = self.v(x).view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
        att = torch.softmax(scores, dim=-1)
        y = att @ v
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.o(y)


class LNAttnLNBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, eps: float = 1e-5):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model, eps=eps)
        self.attn = PureSelfAttention(d_model, n_head)
        self.ln2  = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x)
        x = self.attn(x)
        return self.ln2(x)


class ConvOnlyBlock(nn.Module):
    def __init__(self, Cin, Cout, K=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(Cin, Cout, kernel_size=K,
                              stride=stride, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

# ============================================================
# 벤치 함수 — Mobile 버전
#   모든 bench_* 함수: 모델 빌드 → to_mobile_model → 추론
# ============================================================

@torch.no_grad()
def bench_conv(B=1, Cin=3, Cout=8, H=64, W=64, K=3,
               runs=RUNS, device=DEVICE):
    """단일 Conv2d 레이어 벤치마크 (Mobile)"""
    net = nn.Conv2d(Cin, Cout, K, padding=1, bias=False).to(device)
    x   = torch.randn(B, Cin, H, W, device=device)
    mob = to_mobile_model(net, x, save_name=f"conv_single_{B}_{Cin}_{Cout}")

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = mob(x)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    avg = sum(times) / len(times)

    f = 2 * B * Cin * Cout * K * K * H * W
    b = (B * Cin * H * W + B * Cout * H * W) * 4
    g = f / avg / 1e9
    return avg, f, b, g


@torch.no_grad()
def bench_conv_block(B, Cin, Cout, H, W, K=3, runs=RUNS, device=DEVICE):
    ratios = RATIOS["conv"]

    def build_net():
        layers = []; in_ch = Cin
        for _ in ratios:
            out_ch = max(1, int(Cout))
            layers += [
                nn.Conv2d(in_ch, out_ch, K, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=False),   # 모바일: inplace=False
            ]
            in_ch = out_ch
        return nn.Sequential(*layers).to(device)

    x = torch.randn(B, Cin, H, W, device=device)

    if REBUILD_MODEL_EVERY_RUN:
        times = []
        for _ in range(runs):
            mob = to_mobile_model(build_net(), x)
            times.append(run_model(mob, x, 1))
        avg = sum(times) / len(times)
    else:
        mob = to_mobile_model(build_net(), x, save_name=f"conv_block_{B}_{Cin}_{Cout}")
        avg = run_model(mob, x, runs)

    f = flops_conv3x3(B, Cin, Cout, H, W, K) * len(ratios)
    b = bytes_conv3x3(B, Cin, Cout, H, W, K) * len(ratios)
    g = f / avg / 1e9
    return avg, f, b, g


@torch.no_grad()
def bench_depthwise_block(B, C, H, W, K=3, runs=RUNS, device=DEVICE):
    ratios = RATIOS["dw"]

    def build_net():
        layers = []
        for _ in ratios:
            layers += [
                nn.Conv2d(C, C, K, padding=1, groups=C, bias=False),
                nn.ReLU(inplace=False),
            ]
        return nn.Sequential(*layers).to(device)

    x = torch.randn(B, C, H, W, device=device)

    if REBUILD_MODEL_EVERY_RUN:
        times = []
        for _ in range(runs):
            mob = to_mobile_model(build_net(), x)
            times.append(run_model(mob, x, 1))
        avg = sum(times) / len(times)
    else:
        mob = to_mobile_model(build_net(), x, save_name=f"dw_block_{B}_{C}")
        avg = run_model(mob, x, runs)

    f = flops_dw(B, C, H, W, K) * len(ratios)
    b = bytes_dw(B, C, H, W, K) * len(ratios)
    g = f / avg / 1e9
    return avg, f, b, g


@torch.no_grad()
def bench_gemm_block(M, K, N, runs=RUNS, device=DEVICE):
    ratios = RATIOS["gemm"]

    def build_net():
        layers = []; in_dim = K
        for r in ratios:
            out_dim = max(1, int(N * (0.8 + 0.1*r)))
            layers += [
                nn.Linear(in_dim, out_dim, bias=True),
                nn.ReLU(inplace=False),
                nn.Softmax(dim=-1),
            ]
            in_dim = out_dim
        return nn.Sequential(*layers).to(device)

    x = torch.randn(M, K, device=device)

    if REBUILD_MODEL_EVERY_RUN:
        times = []
        for _ in range(runs):
            mob = to_mobile_model(build_net(), x)
            times.append(run_model(mob, x, 1))
        avg = sum(times) / len(times)
    else:
        mob = to_mobile_model(build_net(), x, save_name=f"gemm_block_{M}_{K}_{N}")
        avg = run_model(mob, x, runs)

    f = flops_gemm(M, K, N) * len(ratios)
    b = bytes_gemm(M, K, N) * len(ratios)
    g = f / avg / 1e9
    return avg, f, b, g


@torch.no_grad()
def bench_attention_block(B, T, H, Dh, runs=RUNS, device=DEVICE):
    ratios = RATIOS["attn"]
    D = H * Dh

    def build_net():
        layers = []
        for _ in ratios:
            layers += [
                PureSelfAttention(d_model=D, n_head=H),
                nn.LayerNorm(D),
                nn.ReLU(inplace=False),
            ]
        return nn.Sequential(*layers).to(device)

    x = torch.randn(B, T, D, device=device)

    if REBUILD_MODEL_EVERY_RUN:
        times = []
        for _ in range(runs):
            mob = to_mobile_model(build_net(), x)
            times.append(run_model(mob, x, 1))
        avg = sum(times) / len(times)
    else:
        mob = to_mobile_model(build_net(), x, save_name=f"attn_block_{B}_{T}_{H}_{Dh}")
        avg = run_model(mob, x, runs)

    f = flops_attn_total(B, T, H, Dh) * len(ratios)
    b = bytes_attn_total(B, T, H, Dh) * len(ratios)
    g = f / avg / 1e9
    return avg, f, b, g


@torch.no_grad()
def bench_ln_attn_ln_block(B, T, H, Dh, runs=RUNS, device=DEVICE):
    ratios = RATIOS["ln_attn_ln"]
    D = H * Dh

    def build_net():
        layers = []
        for _ in ratios:
            layers += [LNAttnLNBlock(d_model=D, n_head=H)]
        return nn.Sequential(*layers).to(device)

    x = torch.randn(B, T, D, device=device)

    if REBUILD_MODEL_EVERY_RUN:
        times = []
        for _ in range(runs):
            mob = to_mobile_model(build_net(), x)
            times.append(run_model(mob, x, 1))
        avg = sum(times) / len(times)
    else:
        mob = to_mobile_model(build_net(), x, save_name=f"ln_attn_ln_{B}_{T}_{H}")
        avg = run_model(mob, x, runs)

    f_ln1, b_ln1 = flops_bytes_layernorm(B, T, D)
    f_attn        = flops_attn_total(B, T, H, Dh)
    b_attn        = bytes_attn_total(B, T, H, Dh)
    f_ln2, b_ln2  = flops_bytes_layernorm(B, T, D)
    f_one = f_ln1 + f_attn + f_ln2
    b_one = b_ln1 + b_attn + b_ln2
    f = f_one * len(ratios)
    b = b_one * len(ratios)
    g = f / avg / 1e9
    return avg, f, b, g

# ============================================================
# 스케일 보간 & 실행
# ============================================================
def scale_linear(v_fit, v_miss, step, steps):
    return int(round(v_fit + (v_miss - v_fit) * (step / (steps - 1))))

def interpolate_cfg(cfg_fit, cfg_miss, step, steps):
    return {k: scale_linear(cfg_fit[k], cfg_miss[k], step, steps) for k in cfg_fit}


def run_scaled_suite():
    ops = [
        #("DEPTH",      bench_depthwise_block,   "dw"),
        ("CONV",       bench_conv_block,         "conv"),
        #("GEMM",       bench_gemm_block,          "gemm"),
        #("ATTN",       bench_attention_block,    "attn"),
        #("LN-ATTN-LN", bench_ln_attn_ln_block,  "ln_attn_ln"),
    ]
    print(f"[PyTorch Mobile]  REBUILD={REBUILD_MODEL_EVERY_RUN}  "
          f"USE_MOBILE_OPTIMIZER={USE_MOBILE_OPTIMIZER}  SAVE_PTL={SAVE_PTL}")
    print(f"{'OP':<12}{'Step':<5}{'GFLOP/s':>10}{'GFLOPs':>10}{'GBytes':>10}{'ms':>12}")
    for name, fn, key in ops:
        for step in range(N_STEPS):
            cfg = interpolate_cfg(SIZES_FIT[key], SIZES_MISS[key], step, N_STEPS)
            avg, f, b, g = fn(**cfg)
            print(f"{name:<12}{step+1:<5}{g:10.2f}{f/1e9:10.3f}"
                  f"{b/1e9:10.3f}{avg*1000:12.2f}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    run_scaled_suite()
