#!/usr/bin/env python3
"""
GEMM Log-Scale Sweep Benchmark
- M, K, N을 로그 스케일로 보간하며 GEMM 성능을 측정
- FLOPs 증가율이 매 스텝 균등하게 유지됨
"""

import torch
import torch.nn as nn
import time

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
DEVICE  = "cpu"
RUNS    = 128
N_STEPS = 10
BYTES_PER = 4

SIZE_FIT  = dict(M=256,  K=256,  N=256)
SIZE_MISS = dict(M=1024, K=1024, N=1024)


# ─────────────────────────────────────────
# 로그 스케일 보간
# ─────────────────────────────────────────
def scale_log(v_fit: int, v_miss: int, step: int, steps: int) -> int:
    """
    step=0 → v_fit, step=steps-1 → v_miss
    중간값은 등비수열: v_fit * (v_miss/v_fit)^(step/(steps-1))
    """
    if v_fit == v_miss or steps == 1:
        return v_fit
    ratio = step / (steps - 1)
    return int(round(v_fit * (v_miss / v_fit) ** ratio))


def interpolate_cfg_log(cfg_fit: dict, cfg_miss: dict, step: int, steps: int) -> dict:
    return {k: scale_log(cfg_fit[k], cfg_miss[k], step, steps) for k in cfg_fit}


# ─────────────────────────────────────────
# FLOPs / Bytes
# ─────────────────────────────────────────
def flops_gemm(M, K, N) -> int:
    return 2 * M * K * N

def bytes_gemm(M, K, N) -> int:
    # A(MxK) + B(KxN) + C(MxN) read + C(MxN) write
    return (M * K + K * N + 2 * M * N) * BYTES_PER

def arith_intensity(M, K, N) -> float:
    return flops_gemm(M, K, N) / bytes_gemm(M, K, N)


# ─────────────────────────────────────────
# 벤치마크 (단일 스텝)
# ─────────────────────────────────────────
@torch.no_grad()
def bench_gemm(M: int, K: int, N: int,
               runs: int = RUNS,
               device: str = DEVICE) -> dict:
    """
    nn.Linear(K→N)을 (M,K) 입력으로 실행, 평균 시간 측정
    Returns: avg_ms, gflops, flops, bytes, intensity
    """
    net = nn.Linear(K, N, bias=False).to(device)
    x   = torch.randn(M, K, device=device)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _  = net(x)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_s  = sum(times) / len(times)
    f      = flops_gemm(M, K, N)
    b      = bytes_gemm(M, K, N)

    return dict(
        M       = M,
        K       = K,
        N       = N,
        avg_ms  = avg_s * 1e3,
        gflops  = f / avg_s / 1e9,
        flops   = f,
        nbytes  = b,
        intensity = arith_intensity(M, K, N),
    )


# ─────────────────────────────────────────
# 로그 스케일 스윕
# ─────────────────────────────────────────
def gemm_log_sweep(size_fit  : dict = SIZE_FIT,
                   size_miss : dict = SIZE_MISS,
                   n_steps   : int  = N_STEPS,
                   runs      : int  = RUNS,
                   device    : str  = DEVICE) -> list[dict]:
    """
    FIT → MISS 를 로그 스케일로 n_steps 단계 스윕
    각 스텝의 결과 dict 리스트를 반환
    """
    results = []
    prev_flops = None
    prev_bytes = None

    print(f"\n{'─'*95}")
    print(f"{'GEMM Log-Scale Sweep':^95}")
    print(f"FIT  : M={size_fit['M']}, K={size_fit['K']}, N={size_fit['N']}")
    print(f"MISS : M={size_miss['M']}, K={size_miss['K']}, N={size_miss['N']}")
    print(f"Steps: {n_steps}  |  Runs/step: {runs}  |  Device: {device}")
    print(f"{'─'*95}")
    print(f"{'Step':>4} {'M':>5} {'K':>5} {'N':>5} "
          f"{'FLOPs(M)':>10} {'FLOPs↑':>8} "
          f"{'Bytes(MB)':>10} {'Bytes↑':>8} "
          f"{'AI':>6} "
          f"{'GFLOP/s':>9} {'ms':>8}")
    print(f"{'─'*95}")

    for step in range(n_steps):
        cfg = interpolate_cfg_log(size_fit, size_miss, step, n_steps)
        r   = bench_gemm(**cfg, runs=runs, device=device)

        # FLOPs 증가율
        if prev_flops is None:
            flop_ratio_str  = "   —"
            byte_ratio_str  = "   —"
        else:
            flop_ratio_str  = f"×{r['flops']  / prev_flops:5.2f}"
            byte_ratio_str  = f"×{r['nbytes'] / prev_bytes:5.2f}"
        prev_flops = r['flops']
        prev_bytes = r['nbytes']

        print(f"{step+1:>4} {r['M']:>5} {r['K']:>5} {r['N']:>5} "
              f"{r['flops']/1e6:>10.1f} {flop_ratio_str:>8} "
              f"{r['nbytes']/1e6:>10.2f} {byte_ratio_str:>8} "
              f"{r['intensity']:>6.2f} "
              f"{r['gflops']:>9.2f} {r['avg_ms']:>8.3f}")

        results.append(r)

    print(f"{'─'*95}")
    print(f"  AI = Arithmetic Intensity (FLOPs/Byte)  |  Bytes = A+B+C 메모리 접근량 (float32)")
    return results


# ─────────────────────────────────────────
# 선형 vs 로그 비교 (참고용)
# ─────────────────────────────────────────
def compare_linear_vs_log(size_fit  : dict = SIZE_FIT,
                           size_miss : dict = SIZE_MISS,
                           n_steps   : int  = N_STEPS):
    """
    실제 실행 없이 FLOPs 분포만 비교 출력
    """
    def scale_linear(v_fit, v_miss, step, steps):
        return int(round(v_fit + (v_miss - v_fit) * (step / (steps - 1))))

    print(f"\n{'─'*65}")
    print(f"{'선형 vs 로그 FLOPs 비교 (M=K=N 동시 변화)':^55}")
    print(f"{'─'*65}")
    print(f"{'Step':>4} {'선형 M':>7} {'선형 FLOPs':>12} {'선형↑':>7} "
          f"{'로그 M':>7} {'로그 FLOPs':>12} {'로그↑':>7}")
    print(f"{'─'*65}")

    prev_lin = prev_log = None
    for step in range(n_steps):
        m_lin = scale_linear(size_fit['M'], size_miss['M'], step, n_steps)
        m_log = scale_log   (size_fit['M'], size_miss['M'], step, n_steps)
        f_lin = flops_gemm(m_lin, m_lin, m_lin)
        f_log = flops_gemm(m_log, m_log, m_log)

        r_lin = f"×{f_lin/prev_lin:.2f}" if prev_lin else "    —"
        r_log = f"×{f_log/prev_log:.2f}" if prev_log else "    —"
        prev_lin, prev_log = f_lin, f_log

        print(f"{step+1:>4} {m_lin:>7} {f_lin/1e6:>11.1f}M {r_lin:>7} "
              f"{m_log:>7} {f_log/1e6:>11.1f}M {r_log:>7}")

    print(f"{'─'*65}")


# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────
if __name__ == "__main__":
    compare_linear_vs_log()
    results = gemm_log_sweep()